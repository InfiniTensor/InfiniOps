import argparse
import json
import pathlib
import textwrap

from _operator_utils import OperatorExtractor, get_all_ops, snake_to_pascal
from _generate_pybind11 import generate_pybind11
from _generate_legacy_c import generate_legacy_c

_GENERATION_DIR = pathlib.Path("generated")

_BINDINGS_DIR = _GENERATION_DIR / "bindings"

_GENERATED_SRC_DIR = _GENERATION_DIR / "src"

_INCLUDE_DIR = _GENERATION_DIR / "include"

_INDENTATION = "  "

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An automatic wrapper generator.")

    parser.add_argument(
        "--devices",
        nargs="+",
        default="cpu",
        type=str,
        help="Devices to use. Please pick from cpu, nvidia, cambricon, ascend, metax, moore, iluvatar, kunlun, hygon, and qy. (default: cpu)",
    )

    args = parser.parse_args()

    _BINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    _GENERATED_SRC_DIR.mkdir(parents=True, exist_ok=True)
    _INCLUDE_DIR.mkdir(parents=True, exist_ok=True)

    ops_json = pathlib.Path("ops.json")

    if ops_json.exists():
        ops = json.loads(ops_json.read_text())
    else:
        ops = get_all_ops(args.devices)

    header_paths = []
    bind_func_names = []

    for op_name, impl_paths in ops.items():
        extractor = OperatorExtractor()
        operator = extractor(op_name)

        source_path = _GENERATED_SRC_DIR / op_name
        header_name = f"{op_name}.h"
        bind_func_name = f"Bind{snake_to_pascal(op_name)}"

        (_BINDINGS_DIR / header_name).write_text(generate_pybind11(operator))

        legacy_c_source, legacy_c_header = generate_legacy_c(operator, impl_paths)
        source_path.mkdir(exist_ok=True)
        (_GENERATED_SRC_DIR / op_name / "operator.cc").write_text(legacy_c_source)
        (_INCLUDE_DIR / header_name).write_text(legacy_c_header)

        header_paths.append(header_name)
        bind_func_names.append(bind_func_name)

    impl_includes = "\n".join(
        f'#include "{impl_path}"'
        for impl_paths in ops.values()
        for impl_path in impl_paths
    )
    op_includes = "\n".join(f'#include "{header_path}"' for header_path in header_paths)
    bind_func_calls = "\n".join(
        f"{bind_func_name}(m);" for bind_func_name in bind_func_names
    )

    (_BINDINGS_DIR / "ops.cc").write_text(f"""#include <pybind11/pybind11.h>

// clang-format off
{impl_includes}
// clang-format on

{op_includes}

namespace infini::ops {{

PYBIND11_MODULE(ops, m) {{
{textwrap.indent(bind_func_calls, _INDENTATION)}
}}

}}  // namespace infini::ops
""")
