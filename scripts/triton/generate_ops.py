import argparse
import importlib.util
import pathlib
import shutil
import sys

import aot

_PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

_KERNEL_NAME = "kernel"
_OPS_DIR = _PROJECT_DIR / "src" / "triton" / "ops"


def _prepend_sys_path(path):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)


def _find_op_modules():
    return {
        path.parent.name: path
        for path in sorted(_OPS_DIR.glob("*/build.py"))
        if path.is_file()
    }


def _build_manifest(output_dir):
    return sorted(str(path) for path in pathlib.Path(output_dir).rglob("*.c"))


def _write_cmake_manifest(output_dir, sources):
    manifest_path = pathlib.Path(output_dir) / "manifest.cmake"
    lines = ["set(INFINIOPS_TRITON_SOURCES"]
    lines.extend(f'    "{source}"' for source in sources)
    lines.append(")")
    lines.append("")
    lines.append(f'set(INFINIOPS_TRITON_INCLUDE_DIRS "{output_dir}")')
    lines.append("")
    manifest_path.write_text("\n".join(lines) + "\n")


def _load_op_module(path):
    _prepend_sys_path(path.parent)
    spec = importlib.util.spec_from_file_location(
        f"infiniops_triton_{path.parent.name}_build",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def generate(ops, *, output_dir):
    op_modules = _find_op_modules()
    unknown_ops = tuple(op for op in ops if op not in op_modules)

    if unknown_ops:
        raise ValueError(f"unsupported Triton ops: {', '.join(unknown_ops)}")

    output_dir = pathlib.Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for op in ops:
        path = op_modules[op]
        kernel_path = path.parent / f"{op}.py"
        module = _load_op_module(path)
        kernel_args = aot.kernel_args(kernel_path, _KERNEL_NAME)
        headers = []
        dispatch_configs = []
        for configs in module.configs():
            out_base = aot.build(
                configs,
                path=kernel_path,
                kernel_name=_KERNEL_NAME,
                out_dir=output_dir / op,
                kernel_args=kernel_args,
            )
            headers.append(out_base.with_suffix(".h"))
            dispatch_configs.append(configs[0])
        aot.write_header(
            headers,
            output_dir / op / f"infini_ops_triton_{op}.h",
            op_name=op,
            configs=dispatch_configs,
            kernel_args=kernel_args,
        )

    sources = _build_manifest(output_dir)
    _write_cmake_manifest(output_dir, sources)

    return sources


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Triton operator sources for InfiniOps."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ops", nargs="+", default=tuple(_find_op_modules()))

    return parser.parse_args()


def main():
    args = _parse_args()
    generate(args.ops, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
