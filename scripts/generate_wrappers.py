import json
import pathlib
import textwrap

import clang.cindex
from clang.cindex import CursorKind

_GENERATION_DIR = pathlib.Path("generated")

_BINDINGS_DIR = _GENERATION_DIR / "bindings"

_INDENTATION = "  "


class _OperatorExtractor:
    def __init__(self, path):
        index = clang.cindex.Index.create()
        args = ("-std=c++17", "-x", "c++", "-I", "src")
        self._translation_unit = index.parse(path, args=args)

    def __call__(self, op_name):
        nodes = tuple(type(self)._find(self._translation_unit.cursor, op_name))

        constructor = None
        call = None

        for node in nodes:
            if node.kind == CursorKind.CONSTRUCTOR:
                constructor = node
            elif node.kind == CursorKind.CXX_METHOD and node.spelling == "operator()":
                call = node

        return _Operator(op_name, constructor, call)

    @staticmethod
    def _find(node, op_name):
        if node.semantic_parent and node.semantic_parent.spelling == "Operator":
            for child in node.semantic_parent.get_children():
                if (
                    child.kind == CursorKind.CXX_BASE_SPECIFIER
                    and child.spelling == op_name
                ):
                    yield node

        for child in node.get_children():
            yield from _OperatorExtractor._find(child, op_name)


class _Operator:
    def __init__(self, name, constructor, call):
        self.name = name

        self.constructor = constructor

        self.call = call


def _generate_pybind11(operator):
    def _generate_params(node):
        return ", ".join(
            f"{arg.type.spelling} {arg.spelling}" for arg in node.get_arguments()
        )

    def _generate_constructor_arguments(node):
        return ", ".join(
            _generate_tensor_caster(arg.spelling)
            if "Tensor" in arg.type.spelling
            else arg.spelling
            for arg in node.get_arguments()
        )

    def _generate_call_arguments(node):
        return ", ".join(
            _generate_data_getter(arg.spelling)
            if arg.spelling != "stream"
            else "reinterpret_cast<void*>(stream)"
            for arg in node.get_arguments()
        )

    def _generate_tensor_caster(name):
        return f'Tensor{{reinterpret_cast<void*>({name}.attr("data_ptr")().cast<std::uintptr_t>()), {name}.attr("shape").cast<Tensor::Shape>(), DataType::FromString(py::str({name}.attr("dtype")).attr("split")(".").attr("__getitem__")(-1).cast<std::string>()), Device{{Device::TypeFromString({name}.attr("device").attr("type").cast<std::string>()), {name}.attr("device").attr("index").is_none() ? 0 : {name}.attr("device").attr("index").cast<int>()}}, {name}.attr("stride")().cast<Tensor::Strides>()}}'

    def _generate_data_getter(name):
        return (
            f'reinterpret_cast<void*>({name}.attr("data_ptr")().cast<std::uintptr_t>())'
        )

    op_name = operator.name

    constructor_params = _generate_params(operator.constructor)
    constructor_params = constructor_params.replace(
        "const Tensor", "py::object"
    ).replace("Tensor", "py::object")

    call_params = _generate_params(operator.call)
    call_params = (
        call_params.replace("void * stream", "std::uintptr_t stream")
        .replace("const void *", "py::object")
        .replace("void *", "py::object")
    )

    return f"""#ifndef INFINI_OPS_BINDINGS_{op_name.upper()}_H_
#define INFINI_OPS_BINDINGS_{op_name.upper()}_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "base/{op_name.lower()}.h"

namespace py = pybind11;

namespace infini::ops {{

void Bind{op_name}(py::module& m) {{
  using Self = {op_name};

  py::class_<Self>(m, "{op_name}")
      .def(py::init([]({constructor_params}) {{
        return std::unique_ptr<Self>{{static_cast<Self*>(Self::make({_generate_constructor_arguments(operator.constructor)}).release())}};
      }}))
      .def("__call__", [](const Self& self, {call_params}) {{
        return self.operator()({_generate_call_arguments(operator.call)});
      }});
}}

}}  // namespace infini::ops

#endif
"""


if __name__ == "__main__":
    _BINDINGS_DIR.mkdir(parents=True, exist_ok=True)

    with open("ops.json") as f:
        ops = json.load(f)

    header_paths = []
    bind_func_names = []

    for op_name, op_path in ops.items():
        extractor = _OperatorExtractor(op_path)
        operator = extractor(op_name)

        header_name = f"{op_name.lower()}.h"
        bind_func_name = f"Bind{op_name}"

        (_BINDINGS_DIR / header_name).write_text(_generate_pybind11(operator))

        header_paths.append(header_name)
        bind_func_names.append(bind_func_name)

    impl_includes = "\n".join(
        f'#include "{header_path}"' for header_path in ops.values()
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
