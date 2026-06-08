import pathlib

from triton.tools.compile import CompileArgs, compile_kernel
from triton.tools import link

_KERNEL_PATH = pathlib.Path(__file__).parent / "add.py"
_KERNEL_NAME = "kernel"
_DTYPES = (
    "fp16",
    "bf16",
    "fp32",
    "fp64",
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
)
_BLOCK_SIZES = (512, 1024)
_NUM_WARPS = 4
_NUM_STAGES = 3


def _compile_variants(variant_dir, dtype):
    out_name = f"infini_ops_triton_add_{dtype}"
    headers = []
    for block_size in _BLOCK_SIZES:
        aligned_sig = (
            f"*{dtype}:16, *{dtype}:16, *{dtype}:16, "
            f"*i64, *i64, *i64, *i64, "
            f"i32, i32, i32, i32, i32, {block_size}"
        )
        _, files = compile_kernel(
            CompileArgs(
                path=str(_KERNEL_PATH),
                kernel_name=_KERNEL_NAME,
                signature=aligned_sig,
                grid=f"(n_elements + {block_size} - 1) / {block_size}, 1, 1",
                num_warps=_NUM_WARPS,
                num_stages=_NUM_STAGES,
                out_name=out_name,
                out_path=variant_dir / out_name,
                target=None,
            )
        )
        headers.extend(f for f in files if f.suffix == ".h")

        generic_sig = (
            f"*{dtype}, *{dtype}, *{dtype}, "
            f"*i64, *i64, *i64, *i64, "
            f"i32, i32, i32, i32, i32, {block_size}"
        )
        _, files = compile_kernel(
            CompileArgs(
                path=str(_KERNEL_PATH),
                kernel_name=_KERNEL_NAME,
                signature=generic_sig,
                grid=f"(n_elements + {block_size} - 1) / {block_size}, 1, 1",
                num_warps=_NUM_WARPS,
                num_stages=_NUM_STAGES,
                out_name=out_name,
                out_path=variant_dir / out_name,
                target=None,
            )
        )
        headers.extend(f for f in files if f.suffix == ".h")
    return headers, out_name


def _link_one_dtype(variant_dir, headers, out_name):
    parser = link.HeaderParser()
    for h in headers:
        parser.extract_linker_meta(h.read_text())

    out_base = variant_dir / out_name
    first_meta = next(iter(parser.kernels.values()))[0]
    backend_prelude = (
        pathlib.Path(link.__file__).parent / "extra" / parser.backend_name / "link.h"
    ).read_text()

    algo_decls = [link.make_algo_decls(name, m) for name, m in parser.kernels.items()]
    out_base.with_suffix(".h").write_text(
        backend_prelude
        + "\n".join(algo_decls)
        + "\n"
        + link.make_get_num_algos_decl(first_meta)
        + "\n"
        + link.make_global_decl(first_meta)
    )
    defs = [
        link.make_kernel_hints_dispatcher(name, m) for name, m in parser.kernels.items()
    ]
    names = list(parser.kernels.keys())
    src = backend_prelude
    src += "#include <stdint.h>\n#include <assert.h>\n\n"
    src += "\n".join(defs) + "\n"
    src += link.make_func_pointers(names, first_meta) + "\n"
    src += link.make_get_num_algos_def(first_meta) + "\n"
    src += link.make_kernel_meta_const_dispatcher(first_meta) + "\n"
    src += link.make_kernel_load_def(names, first_meta) + "\n"
    src += link.make_default_algo_kernel(first_meta)
    out_base.with_suffix(".c").write_text(src)


def build(output_dir: pathlib.Path):
    variant_dir = output_dir / "add"
    variant_dir.mkdir(parents=True, exist_ok=True)

    for dtype in _DTYPES:
        headers, out_name = _compile_variants(variant_dir, dtype)
        _link_one_dtype(variant_dir, headers, out_name)
