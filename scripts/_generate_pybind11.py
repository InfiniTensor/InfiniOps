from _operator_utils import snake_to_pascal


def generate_pybind11(operator):
    def _generate_params(node):
        return (
            ", ".join(
                f"{arg.type.spelling} {arg.spelling}"
                for arg in node.get_arguments()
                if arg.spelling != "stream"
            )
            .replace("const Tensor", "py::object")
            .replace("Tensor", "py::object")
        )

    def _generate_arguments(node):
        return ", ".join(
            f"TensorFromPybind11Handle({arg.spelling})"
            if "Tensor" in arg.type.spelling
            else arg.spelling
            for arg in node.get_arguments()
            if arg.spelling != "stream"
        )

    op_name = operator.name

    def _generate_init(constructor):
        constructor_params = _generate_params(constructor)

        return f"""      .def(py::init([]({constructor_params}) {{
        return std::unique_ptr<Self>{{static_cast<Self*>(Self::make({_generate_arguments(constructor)}).release())}};
      }}))"""

    def _generate_py_args(node):
        return ", ".join(
            f'py::arg("{arg.spelling}")'
            for arg in node.get_arguments()
            if arg.spelling != "stream"
        )

    def _generate_call(op_name, call, method=True):
        call_params = _generate_params(call)
        call_args = _generate_arguments(call)

        if not method:
            params = (
                f"{call_params}, std::size_t implementation_index"
                if call_params
                else "std::size_t implementation_index"
            )
            py_args = _generate_py_args(call)
            py_args_str = f"{py_args}, " if py_args else ""

            return f"""  m.def("{op_name}", []({params}) {{
    Config config;
    config.set_implementation_index(implementation_index);
    return Self::call({{}}, config, {call_args});
  }}, {py_args_str}py::kw_only(), py::arg("implementation_index") = 0);"""

        return f"""      .def("__call__", [](const Self& self, {call_params}) {{
        return static_cast<const Operator<Self>&>(self)({call_args});
      }})"""

    inits = "\n".join(
        _generate_init(constructor) for constructor in operator.constructors
    )
    calls = "\n".join(_generate_call(operator.name, call) for call in operator.calls)
    callers = "\n".join(
        _generate_call(operator.name, call, method=False) for call in operator.calls
    )

    pascal_case_op_name = snake_to_pascal(op_name)

    return f"""#ifndef INFINI_OPS_BINDINGS_{op_name.upper()}_H_
#define INFINI_OPS_BINDINGS_{op_name.upper()}_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "base/{op_name}.h"
#include "config.h"
#include "pybind11_utils.h"

namespace py = pybind11;

namespace infini::ops {{

void Bind{pascal_case_op_name}(py::module& m) {{
  using Self = {pascal_case_op_name};

  py::class_<Self>(m, "{pascal_case_op_name}")
{inits}
{calls}
      .def_static("active_implementation_indices", [](const std::string& device) {{
        return Self::active_implementation_indices(DeviceTypeFromString(device));
      }});

{callers}
}}

}}  // namespace infini::ops

#endif
"""
