import argparse
import json
import pathlib
import re
import shutil
import subprocess
import textwrap

import clang.cindex
from clang.cindex import CursorKind

_SRC_DIR = pathlib.Path("src")

_BASE_DIR = _SRC_DIR / "base"

_GENERATION_DIR = pathlib.Path("generated")

_BINDINGS_DIR = _GENERATION_DIR / "bindings"

_GENERATED_SRC_DIR = _GENERATION_DIR / "src"

_INCLUDE_DIR = _GENERATION_DIR / "include"

_INDENTATION = "  "


class _OperatorExtractor:
    def __call__(self, op_name):
        def _get_system_include_flags():
            def _get_compilers():
                compilers = []

                for compiler in ("clang++", "g++"):
                    if shutil.which(compiler) is not None:
                        compilers.append(compiler)

                return compilers

            system_include_flags = []

            for compiler in _get_compilers():
                for line in subprocess.getoutput(
                    f"{compiler} -E -x c++ -v /dev/null"
                ).splitlines():
                    if not line.startswith(" "):
                        continue

                    system_include_flags.append("-isystem")
                    system_include_flags.append(line.strip())

            return system_include_flags

        system_include_flags = _get_system_include_flags()

        index = clang.cindex.Index.create()
        args = ("-std=c++17", "-x", "c++", "-I", "src") + tuple(system_include_flags)
        translation_unit = index.parse(f"src/base/{op_name}.h", args=args)

        nodes = tuple(type(self)._find(translation_unit.cursor, op_name))

        constructors = []
        calls = []

        for node in nodes:
            if node.kind == CursorKind.CONSTRUCTOR:
                constructors.append(node)
            elif node.kind == CursorKind.CXX_METHOD and node.spelling == "operator()":
                calls.append(node)

        return _Operator(op_name, constructors, calls)

    @staticmethod
    def _find(node, op_name):
        pascal_case_op_name = _snake_to_pascal(op_name)

        if (
            node.semantic_parent
            and node.semantic_parent.spelling == pascal_case_op_name
        ):
            yield node

        for child in node.get_children():
            yield from _OperatorExtractor._find(child, op_name)


class _Operator:
    def __init__(self, name, constructors, calls):
        self.name = name

        self.constructors = constructors

        self.calls = calls


def _find_optional_tensor_params(op_name):
    """Return a set of parameter names declared as `std::optional<Tensor>` in
    the base header. `libclang` resolves the type to `int` when the STL
    headers are not fully available, so we fall back to a regex scan of the
    source text.
    """
    source = (_BASE_DIR / f"{op_name}.h").read_text()

    return set(re.findall(r"std::optional<Tensor>\s+(\w+)", source))


def _generate_pybind11(operator):
    optional_tensor_params = _find_optional_tensor_params(operator.name)

    def _is_optional_tensor(arg):
        if arg.spelling in optional_tensor_params:
            return True

        return "std::optional" in arg.type.spelling and "Tensor" in arg.type.spelling

    def _generate_params(node):
        parts = []

        for arg in node.get_arguments():
            if arg.spelling == "stream":
                continue

            if _is_optional_tensor(arg):
                parts.append(f"std::optional<py::object> {arg.spelling}")
            else:
                param = arg.type.spelling.replace("const Tensor", "py::object").replace(
                    "Tensor", "py::object"
                )
                parts.append(f"{param} {arg.spelling}")

        return ", ".join(parts)

    def _generate_arguments(node):
        args = []

        for arg in node.get_arguments():
            if arg.spelling == "stream":
                continue

            if _is_optional_tensor(arg):
                args.append(f"OptionalTensorFromPybind11Handle({arg.spelling})")
            elif "Tensor" in arg.type.spelling:
                args.append(f"TensorFromPybind11Handle({arg.spelling})")
            else:
                args.append(arg.spelling)

        return ", ".join(args)

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
                f"{call_params}, std::uintptr_t stream, std::size_t implementation_index"
                if call_params
                else "std::uintptr_t stream, std::size_t implementation_index"
            )
            py_args = _generate_py_args(call)
            py_args_str = f"{py_args}, " if py_args else ""

            return (
                f'  m.def("{op_name}", []({params}) {{\n'
                f"    Handle handle;\n"
                f"    if (stream) {{\n"
                f"      handle.set_stream(reinterpret_cast<void*>(stream));\n"
                f"    }}\n"
                f"    Config config;\n"
                f"    config.set_implementation_index(implementation_index);\n"
                f"    return Self::call(handle, config, {call_args});\n"
                f'  }}, {py_args_str}py::kw_only(), py::arg("stream") = 0, py::arg("implementation_index") = 0);'
            )

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

    pascal_case_op_name = _snake_to_pascal(op_name)

    return f"""#ifndef INFINI_OPS_BINDINGS_{op_name.upper()}_H_
#define INFINI_OPS_BINDINGS_{op_name.upper()}_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "base/{op_name}.h"
#include "config.h"
#include "handle.h"
#include "operator.h"
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


def _generate_legacy_c(operator, paths):
    def _generate_source(operator):
        impl_includes = "\n".join(
            f'#include "{str(path).removeprefix("src/")}"' for path in paths
        )

        return f"""#include "../../handle.h"
#include "../../tensor.h"
#include "infiniop/ops/{operator.name.lower()}.h"
{impl_includes}

static infini::ops::DataType DataTypeFromInfiniDType(
    const infiniDtype_t& dtype) {{
  static constexpr infini::ops::ConstexprMap<infiniDtype_t,
                                             infini::ops::DataType, 12>
      kInfiniDTypeToDataType{{
          {{{{{{INFINI_DTYPE_I8, infini::ops::DataType::kInt8}},
            {{INFINI_DTYPE_I16, infini::ops::DataType::kInt16}},
            {{INFINI_DTYPE_I32, infini::ops::DataType::kInt32}},
            {{INFINI_DTYPE_I64, infini::ops::DataType::kInt64}},
            {{INFINI_DTYPE_U8, infini::ops::DataType::kUInt8}},
            {{INFINI_DTYPE_U16, infini::ops::DataType::kUInt16}},
            {{INFINI_DTYPE_U32, infini::ops::DataType::kUInt32}},
            {{INFINI_DTYPE_U64, infini::ops::DataType::kUInt64}},
            {{INFINI_DTYPE_F16, infini::ops::DataType::kFloat16}},
            {{INFINI_DTYPE_BF16, infini::ops::DataType::kBFloat16}},
            {{INFINI_DTYPE_F32, infini::ops::DataType::kFloat32}},
            {{INFINI_DTYPE_F64, infini::ops::DataType::kFloat64}}}}}}}};

  return kInfiniDTypeToDataType.at(dtype);
}}

static infini::ops::Device::Type DeviceTypeFromInfiniDevice(
    const infiniDevice_t& device) {{
  static constexpr infini::ops::ConstexprMap<
      infiniDevice_t, infini::ops::Device::Type,
      static_cast<std::size_t>(INFINI_DEVICE_TYPE_COUNT)>
      kInfiniDeviceToDeviceType{{
          {{{{{{INFINI_DEVICE_CPU, infini::ops::Device::Type::kCpu}},
            {{INFINI_DEVICE_NVIDIA, infini::ops::Device::Type::kNvidia}},
            {{INFINI_DEVICE_CAMBRICON, infini::ops::Device::Type::kCambricon}},
            {{INFINI_DEVICE_ASCEND, infini::ops::Device::Type::kAscend}},
            {{INFINI_DEVICE_METAX, infini::ops::Device::Type::kMetax}},
            {{INFINI_DEVICE_MOORE, infini::ops::Device::Type::kMoore}},
            {{INFINI_DEVICE_ILUVATAR, infini::ops::Device::Type::kIluvatar}},
            {{INFINI_DEVICE_KUNLUN, infini::ops::Device::Type::kKunlun}},
            {{INFINI_DEVICE_HYGON, infini::ops::Device::Type::kHygon}},
            {{INFINI_DEVICE_QY, infini::ops::Device::Type::kQy}}}}}}}};

  return kInfiniDeviceToDeviceType.at(device);
}}

__C {_generate_create_func_def(operator)}

__C {_generate_get_workspace_size_func_def(operator)}

__C {_generate_call_func_def(operator)}

__C {_generate_destroy_func_def(operator)}
"""

    def _generate_header(operator):
        return f"""#ifndef __INFINIOP_{operator.name.upper()}_API_H__
#define __INFINIOP_{operator.name.upper()}_API_H__

#include "base/{operator.name.lower()}.h"

typedef struct infini::ops::Operator<infini::ops::{operator.name}> *infiniop{operator.name}Descriptor_t;

__C __export {_generate_create_func_decl(operator)};

__C __export {_generate_get_workspace_size_func_decl(operator)};

__C __export {_generate_call_func_decl(operator)};

__C __export {_generate_destroy_func_decl(operator)};

#endif
"""

    def _generate_create_func_def(operator):
        name = operator.name
        constructor = operator.constructors[-1]

        return f"""{_generate_create_func_decl(operator)} {{
    *desc_ptr = infini::ops::Operator<infini::ops::{name}>::make({_generate_arguments(constructor)}).release();

    return INFINI_STATUS_SUCCESS;
}}"""

    def _generate_get_workspace_size_func_def(operator):
        return f"""{_generate_get_workspace_size_func_decl(operator)} {{
    *size = 0;  // desc->workspace_size();

    return INFINI_STATUS_SUCCESS;
}}"""

    def _generate_call_func_def(operator):
        call = operator.calls[-1]

        return f"""{_generate_call_func_decl(operator)} {{
    (*desc)(stream, {_generate_arguments(call, is_data=True)});

    return INFINI_STATUS_SUCCESS;
}}"""

    def _generate_destroy_func_def(operator):
        return f"""{_generate_destroy_func_decl(operator)} {{
    delete desc;

    return INFINI_STATUS_SUCCESS;
}}"""

    def _generate_create_func_decl(operator):
        name = operator.name
        constructor = operator.constructors[-1]
        params = _generate_params(constructor)

        return f"infiniStatus_t infiniopCreate{name}Descriptor(infiniopHandle_t handle, infiniop{name}Descriptor_t *desc_ptr, {params})"

    def _generate_get_workspace_size_func_decl(operator):
        name = operator.name

        return f"infiniStatus_t infiniopGet{name}WorkspaceSize(infiniop{name}Descriptor_t desc, size_t *size)"

    def _generate_call_func_decl(operator):
        name = operator.name
        call = operator.calls[-1]
        params = _generate_params(call, call=True)
        params = params.replace("void * stream, ", "")

        return f"infiniStatus_t infiniop{name}(infiniop{name}Descriptor_t desc, void *workspace, size_t workspace_size, {params}, void *stream)"

    def _generate_destroy_func_decl(operator):
        name = operator.name

        return f"infiniStatus_t infiniopDestroy{name}Descriptor(infiniop{name}Descriptor_t desc)"

    def _generate_params(node, call=False):
        arguments = tuple(node.get_arguments())

        arguments = (arguments[-1], *arguments[:-1])

        def _handle_tensor(spelling):
            if call:
                return spelling.replace("Tensor", "void *")
            return spelling.replace("Tensor", "infiniopTensorDescriptor_t")

        def _handle_std_optional(spelling):
            return spelling.replace("std::optional<", "").replace(">", "")

        return ", ".join(
            f"{_handle_std_optional(_handle_tensor(arg.type.spelling))} {arg.spelling}"
            for arg in arguments
        )

    def _generate_arguments(node, is_data=False):
        return ", ".join(
            _generate_tensor_caster(arg.spelling, is_data=is_data)
            if "Tensor" in arg.type.spelling
            else arg.spelling
            for arg in node.get_arguments()
            if arg.spelling != "handle" and arg.spelling != "stream"
        )

    def _generate_tensor_caster(name, is_data=False):
        if is_data:
            return f"infini::ops::Tensor(const_cast<void *>({name}), infini::ops::Tensor::Shape{{}})"

        return f"infini::ops::Tensor{{nullptr, {name}->shape(), DataTypeFromInfiniDType({name}->dtype()), infini::ops::Device{{DeviceTypeFromInfiniDevice(handle->device), handle->device_id}}, {name}->strides()}}"

    return _generate_source(operator), _generate_header(operator)


def _snake_to_pascal(snake_str):
    return "".join(word.capitalize() for word in snake_str.split("_"))


def _get_all_ops(devices, with_torch=False):
    scan_dirs = set(devices)

    if with_torch:
        scan_dirs.add("torch")

    ops = {}

    for file_path in _BASE_DIR.iterdir():
        if not file_path.is_file():
            continue

        op_name = file_path.stem

        ops[op_name] = []

        for file_path in _SRC_DIR.rglob("*"):
            if not file_path.is_file() or file_path.parent.parent.name not in scan_dirs:
                continue

            if f"class Operator<{_snake_to_pascal(op_name)}" in file_path.read_text():
                ops[op_name].append(file_path)

    return ops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An automatic wrapper generator.")

    parser.add_argument(
        "--devices",
        nargs="+",
        default="cpu",
        type=str,
        help="Devices to use. Please pick from `cpu`, `nvidia`, `cambricon`, `ascend`, `metax`, `moore`, `iluvatar`, `kunlun`, `hygon`, and `qy`. (default: `cpu`)",
    )

    parser.add_argument(
        "--with-torch",
        action="store_true",
        help="Include PyTorch C++ backend implementations.",
    )

    args = parser.parse_args()

    _BINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    _GENERATED_SRC_DIR.mkdir(parents=True, exist_ok=True)
    _INCLUDE_DIR.mkdir(parents=True, exist_ok=True)

    ops_json = pathlib.Path("ops.json")

    if ops_json.exists():
        ops = json.loads(ops_json.read_text())
    else:
        ops = _get_all_ops(args.devices, with_torch=args.with_torch)

    header_paths = []
    bind_func_names = []

    for op_name, impl_paths in ops.items():
        extractor = _OperatorExtractor()
        operator = extractor(op_name)

        source_path = _GENERATED_SRC_DIR / op_name
        header_name = f"{op_name}.h"
        bind_func_name = f"Bind{_snake_to_pascal(op_name)}"

        (_BINDINGS_DIR / header_name).write_text(_generate_pybind11(operator))

        legacy_c_source, legacy_c_header = _generate_legacy_c(operator, impl_paths)
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
