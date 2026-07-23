import ast
from dataclasses import dataclass
import pathlib
from typing import Any, Sequence

from triton.tools import link
from triton.tools.compile import CompileArgs, compile_kernel


@dataclass(frozen=True)
class Signature:
    pointer_dtypes: dict[str, str]
    pointer_alignments: dict[str, int | None] | None = None
    scalar_dtypes: dict[str, str] | None = None
    constexprs: dict[str, Any] | None = None


@dataclass(frozen=True)
class CompileConfig:
    signature: Signature
    grid: str
    out_name: str
    num_warps: int = 4
    num_stages: int = 3
    target: Any = None


def compile(
    config: CompileConfig,
    *,
    path: pathlib.Path,
    kernel_name: str,
    out_dir: pathlib.Path,
    kernel_args: Sequence[str],
) -> list[pathlib.Path]:
    _, files = compile_kernel(
        CompileArgs(
            path=str(path),
            kernel_name=kernel_name,
            signature=_render_signature(config.signature, kernel_args),
            grid=config.grid,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
            out_name=config.out_name,
            out_path=out_dir / config.out_name,
            target=config.target,
        )
    )

    return [path for path in files if path.suffix == ".h"]


def link_headers(headers: Sequence[pathlib.Path], out_base: pathlib.Path):
    parser = link.HeaderParser()
    for header in headers:
        parser.extract_linker_meta(header.read_text())

    first_meta = next(iter(parser.kernels.values()))[0]
    backend_prelude = (
        pathlib.Path(link.__file__).parent / "extra" / parser.backend_name / "link.h"
    ).read_text()

    out_base.with_suffix(".h").write_text(
        backend_prelude
        + "\n".join(
            link.make_algo_decls(name, meta) for name, meta in parser.kernels.items()
        )
        + "\n"
        + link.make_get_num_algos_decl(first_meta)
        + "\n"
        + link.make_global_decl(first_meta)
    )

    names = list(parser.kernels)
    defs = [
        link.make_kernel_hints_dispatcher(name, meta)
        for name, meta in parser.kernels.items()
    ]

    out_base.with_suffix(".c").write_text(
        backend_prelude
        + "#include <stdint.h>\n#include <assert.h>\n\n"
        + "\n".join(defs)
        + "\n"
        + link.make_func_pointers(names, first_meta)
        + "\n"
        + link.make_get_num_algos_def(first_meta)
        + "\n"
        + link.make_kernel_meta_const_dispatcher(first_meta)
        + "\n"
        + link.make_kernel_load_def(names, first_meta)
        + "\n"
        + link.make_default_algo_kernel(first_meta)
    )


def build(
    configs: Sequence[CompileConfig],
    *,
    path: pathlib.Path,
    kernel_name: str,
    out_dir: pathlib.Path,
    kernel_args: Sequence[str],
) -> pathlib.Path:
    if not configs:
        raise ValueError("empty compile configs")

    out_name = configs[0].out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = []
    for config in configs:
        headers.extend(
            compile(
                config,
                path=path,
                kernel_name=kernel_name,
                out_dir=out_dir,
                kernel_args=kernel_args,
            )
        )

    if not headers:
        raise ValueError(f"no headers generated for {out_name}")

    out_base = out_dir / out_name
    link_headers(headers, out_base)

    return out_base


def write_header(
    headers: Sequence[pathlib.Path],
    out_path: pathlib.Path,
    *,
    op_name: str,
    configs: Sequence[CompileConfig],
    kernel_args: Sequence[str],
):
    guard = f"INFINI_OPS_GENERATED_{out_path.stem.upper()}_H_"
    includes = "\n".join(f'#include "{header.name}"' for header in headers)
    params = _dispatch_params(configs[0].signature, kernel_args)
    param_decls = ", ".join(f"{ty} {name}" for ty, name in params)
    param_names = ", ".join(name for _, name in params)

    body = f"#ifndef {guard}\n#define {guard}\n\n"
    body += f'extern "C" {{\n{includes}\n}}\n\n'
    body += '#include <mutex>\n\n#include "data_type.h"\n\n'
    body += "namespace infini::ops {\n\n"

    body += f"inline TT_ResultTy launch_infini_ops_triton_{op_name}(\n"
    body += f"    DataType dtype, TT_StreamTy stream, {param_decls}) {{\n"
    body += "  switch (dtype) {\n"
    for config in configs:
        dtype = _out_dtype(config.out_name)
        body += f"    case DataType::{_data_type(dtype)}:\n"
        body += f"      return {config.out_name}_default(stream, {param_names});\n"
    body += "    default:\n      return TT_ERROR_INVALID_VALUE;\n  }\n}\n\n"

    body += f"inline void load_infini_ops_triton_{op_name}(DataType dtype) {{\n"
    body += "  switch (dtype) {\n"
    for config in configs:
        dtype = _out_dtype(config.out_name)
        body += f"    case DataType::{_data_type(dtype)}: {{\n"
        body += "      static std::once_flag once;\n"
        body += f"      std::call_once(once, &load_{config.out_name});\n"
        body += "      return;\n    }\n"
    body += "    default:\n      return;\n  }\n}\n\n"

    body += "}  // namespace infini::ops\n\n#endif\n"
    out_path.write_text(body)


def kernel_args(path, kernel_name):
    tree = ast.parse(pathlib.Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == kernel_name:
            return tuple(arg.arg for arg in node.args.args)
    raise ValueError(f"kernel {kernel_name} not found in {path}")


def _render_signature(signature: Signature, args: Sequence[str]) -> str:
    pointer_alignments = signature.pointer_alignments or {}
    scalar_dtypes = signature.scalar_dtypes or {}
    constexprs = signature.constexprs or {}

    parts = []
    for arg in args:
        if arg in constexprs:
            parts.append(str(constexprs[arg]))
        elif arg in signature.pointer_dtypes:
            parts.append(
                _ptr(signature.pointer_dtypes[arg], pointer_alignments.get(arg))
            )
        elif arg in scalar_dtypes:
            parts.append(str(scalar_dtypes[arg]))
        else:
            raise ValueError(f"missing signature rule for {arg}")

    return ", ".join(parts)


def _dispatch_params(signature: Signature, args: Sequence[str]):
    scalar_dtypes = signature.scalar_dtypes or {}
    constexprs = signature.constexprs or {}

    params = []
    for arg in args:
        if arg in constexprs:
            continue
        if arg in signature.pointer_dtypes:
            params.append(("CUdeviceptr", arg))
        elif arg in scalar_dtypes:
            params.append((_scalar_ctype(scalar_dtypes[arg]), arg))
        else:
            raise ValueError(f"missing dispatch rule for {arg}")
    return params


def _scalar_ctype(dtype):
    return {
        "i32": "int32_t",
        "i64": "int64_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp32": "float",
        "fp64": "double",
    }[dtype]


def _out_dtype(out_name):
    return out_name.rsplit("_", 1)[1]


def _data_type(dtype):
    return {
        "fp16": "kFloat16",
        "bf16": "kBFloat16",
        "fp32": "kFloat32",
        "fp64": "kFloat64",
        "i8": "kInt8",
        "i16": "kInt16",
        "i32": "kInt32",
        "i64": "kInt64",
        "u8": "kUInt8",
        "u16": "kUInt16",
        "u32": "kUInt32",
        "u64": "kUInt64",
    }[dtype]


def _ptr(dtype, alignment=None):
    return f"*{dtype}" if alignment is None else f"*{dtype}:{alignment}"
