import json
import pathlib
import textwrap

import clang.cindex
from clang.cindex import CursorKind

_GENERATION_DIR = pathlib.Path("generated")

_BINDINGS_DIR = _GENERATION_DIR / "bindings"

_SRC_DIR = _GENERATION_DIR / "src"

_INCLUDE_DIR = _GENERATION_DIR / "include"

_INDENTATION = "  "


class _OperatorExtractor:
    def __call__(self, op_name):
        index = clang.cindex.Index.create()
        args = ("-std=c++17", "-x", "c++", "-I", "src")
        translation_unit = index.parse(f"src/base/{op_name.lower()}.h", args=args)

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
        if node.semantic_parent and node.semantic_parent.spelling == op_name:
            yield node

        for child in node.get_children():
            yield from _OperatorExtractor._find(child, op_name)


class _Operator:
    def __init__(self, name, constructors, calls):
        self.name = name

        self.constructors = constructors

        self.calls = calls


def _generate_pybind11(operator):
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
            _generate_tensor_caster(arg.spelling)
            if "Tensor" in arg.type.spelling
            else arg.spelling
            for arg in node.get_arguments()
            if arg.spelling != "stream"
        )

    def _generate_tensor_caster(name):
        return f'Tensor{{reinterpret_cast<void*>({name}.attr("data_ptr")().cast<std::uintptr_t>()), {name}.attr("shape").cast<Tensor::Shape>(), DataTypeFromString(py::str({name}.attr("dtype")).attr("split")(".").attr("__getitem__")(-1).cast<std::string>()), Device{{DeviceTypeFromString({name}.attr("device").attr("type").cast<std::string>()), {name}.attr("device").attr("index").is_none() ? 0 : {name}.attr("device").attr("index").cast<int>()}}, {name}.attr("stride")().cast<Tensor::Strides>()}}'

    op_name = operator.name

    def _generate_init(constructor):
        constructor_params = _generate_params(constructor)

        return f"""      .def(py::init([]({constructor_params}) {{
        return std::unique_ptr<Self>{{static_cast<Self*>(Self::make({_generate_arguments(constructor)}).release())}};
      }}))"""

    def _generate_call(op_name, call, method=True):
        call_params = _generate_params(call)

        if not method:
            return f"""  m.def("{op_name.lower()}", []({call_params}) {{ return Self::call({_generate_arguments(call)}); }});"""

        return f"""      .def("__call__", [](const Self& self, {call_params}) {{
        return static_cast<const Operator<Self>&>(self)({_generate_arguments(call)});
      }})"""

    inits = "\n".join(
        _generate_init(constructor) for constructor in operator.constructors
    )
    calls = "\n".join(_generate_call(operator.name, call) for call in operator.calls)
    callers = "\n".join(
        _generate_call(operator.name, call, method=False) for call in operator.calls
    )

    return f"""#ifndef INFINI_OPS_BINDINGS_{op_name.upper()}_H_
#define INFINI_OPS_BINDINGS_{op_name.upper()}_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <unordered_map>

#include "base/{op_name.lower()}.h"

namespace py = pybind11;

namespace infini::ops {{

inline DataType DataTypeFromString(const std::string& name) {{
  return kStringToDataType.at(name);
}}

inline Device::Type DeviceTypeFromString(const std::string& name) {{
  static const std::unordered_map<std::string, Device::Type> kTorchNameToTypes{{
      {{"cpu", Device::Type::kCpu}},
#ifdef USE_NVIDIA
      {{"cuda", Device::Type::kNvidia}},
#endif
#ifdef USE_METAX
      {{"cuda", Device::Type::kMetax}},
#endif
#ifdef USE_ILUVATAR
      {{"cuda", Device::Type::kIluvatar}},
#endif
#ifdef USE_KUNLUN
      {{"cuda", Device::Type::kKunlun}},
#endif
#ifdef USE_HYGON
      {{"cuda", Device::Type::kHygon}},
#endif
#ifdef USE_QY
      {{"cuda", Device::Type::kQy}},
#endif
      {{"mlu", Device::Type::kCambricon}}, {{"npu", Device::Type::kAscend}},
      {{"musa", Device::Type::kMoore}}}};

  auto it{{kTorchNameToTypes.find(name)}};

  if (it != kTorchNameToTypes.cend()) {{
    return it->second;
  }}

  return Device::TypeFromString(name);
}}

void Bind{op_name}(py::module& m) {{
  using Self = {op_name};

  py::class_<Self>(m, "{op_name}")
{inits}
{calls};

{callers}
}}

}}  // namespace infini::ops

#endif
"""


def _generate_legacy_c(operator, paths):
    def _generate_source(operator):
        impl_includes = "\n".join(
            f'#include "{path.removeprefix("src/")}"' for path in paths
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


if __name__ == "__main__":
    _BINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    _SRC_DIR.mkdir(parents=True, exist_ok=True)
    _INCLUDE_DIR.mkdir(parents=True, exist_ok=True)

    with open("ops.json") as f:
        ops = json.load(f)

    header_paths = []
    bind_func_names = []

    for op_name, impl_paths in ops.items():
        extractor = _OperatorExtractor()
        operator = extractor(op_name)

        source_path = _SRC_DIR / op_name.lower()
        header_name = f"{op_name.lower()}.h"
        bind_func_name = f"Bind{op_name}"

        (_BINDINGS_DIR / header_name).write_text(_generate_pybind11(operator))

        legacy_c_source, legacy_c_header = _generate_legacy_c(operator, impl_paths)
        source_path.mkdir(exist_ok=True)
        (_SRC_DIR / op_name.lower() / "operator.cc").write_text(legacy_c_source)
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
{_INDENTATION}m.def("set_stream", [](std::uintptr_t stream) {{ OperatorBase::set_stream(reinterpret_cast<void*>(stream)); }});
{textwrap.indent(bind_func_calls, _INDENTATION)}
}}

}}  // namespace infini::ops
""")
