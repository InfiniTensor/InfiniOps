from _operator_utils import snake_to_pascal


def generate_legacy_c(operator, paths):
    pascal_name = snake_to_pascal(operator.name)

    # Map InfiniOps device directory names to InfiniCore preprocessor guards.
    _DEVICE_GUARDS = {
        "cpu": "ENABLE_CPU_API",
        "nvidia": "ENABLE_NVIDIA_API",
        "cambricon": "ENABLE_CAMBRICON_API",
        "ascend": "ENABLE_ASCEND_API",
        "metax": "ENABLE_METAX_API",
        "moore": "ENABLE_MOORE_API",
        "iluvatar": "ENABLE_ILUVATAR_API",
        "kunlun": "ENABLE_KUNLUN_API",
        "hygon": "ENABLE_HYGON_API",
        "qy": "ENABLE_QY_API",
    }

    def _generate_guarded_includes():
        lines = []

        for path in paths:
            rel = str(path).removeprefix("src/")
            device = rel.split("/")[0]
            guard = _DEVICE_GUARDS.get(device)

            if guard:
                lines.append(f"#ifdef {guard}")
                lines.append(f'#include "{rel}"')
                lines.append("#endif")
            else:
                lines.append(f'#include "{rel}"')

        return "\n".join(lines)

    def _generate_source(operator):
        impl_includes = _generate_guarded_includes()

        return f"""#include "../../handle.h"
#include "../../tensor.h"
#include "infiniop/ops/{operator.name}.h"
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

__INFINI_C {_generate_create_func_def(operator)}

__INFINI_C {_generate_get_workspace_size_func_def(operator)}

__INFINI_C {_generate_call_func_def(operator)}

__INFINI_C {_generate_destroy_func_def(operator)}
"""

    def _generate_header(operator):
        return f"""#ifndef __INFINIOP_{operator.name.upper()}_API_H__
#define __INFINIOP_{operator.name.upper()}_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniop{pascal_name}Descriptor_t;

__INFINI_C __export {_generate_create_func_decl(operator)};

__INFINI_C __export {_generate_get_workspace_size_func_decl(operator)};

__INFINI_C __export {_generate_call_func_decl(operator)};

__INFINI_C __export {_generate_destroy_func_decl(operator)};

#endif
"""

    def _generate_create_func_def(operator):
        constructor = operator.constructors[-1]

        return f"""{_generate_create_func_decl(operator)} {{
    *desc_ptr = reinterpret_cast<infiniop{pascal_name}Descriptor_t>(infini::ops::Operator<infini::ops::{pascal_name}>::make({_generate_arguments(constructor)}).release());

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
    auto *op = reinterpret_cast<infini::ops::Operator<infini::ops::{pascal_name}> *>(desc);
    op->set_stream(stream);
    op->set_workspace(workspace);
    op->set_workspace_size_in_bytes(workspace_size);
    const auto &op_ref = *op;
    op_ref({_generate_arguments(call, is_data=True)});

    return INFINI_STATUS_SUCCESS;
}}"""

    def _generate_destroy_func_def(operator):
        return f"""{_generate_destroy_func_decl(operator)} {{
    delete reinterpret_cast<infini::ops::Operator<infini::ops::{pascal_name}> *>(desc);

    return INFINI_STATUS_SUCCESS;
}}"""

    def _generate_create_func_decl(operator):
        constructor = operator.constructors[-1]
        params = _generate_params(constructor)

        return f"infiniStatus_t infiniopCreate{pascal_name}Descriptor(infiniopHandle_t handle, infiniop{pascal_name}Descriptor_t *desc_ptr, {params})"

    def _generate_get_workspace_size_func_decl(operator):
        return f"infiniStatus_t infiniopGet{pascal_name}WorkspaceSize(infiniop{pascal_name}Descriptor_t desc, size_t *size)"

    def _generate_call_func_decl(operator):
        call = operator.calls[-1]
        params = _generate_params(call, call=True)
        params = params.replace("void * stream, ", "")

        return f"infiniStatus_t infiniop{pascal_name}(infiniop{pascal_name}Descriptor_t desc, void *workspace, size_t workspace_size, {params}, void *stream)"

    def _generate_destroy_func_decl(operator):
        return f"infiniStatus_t infiniopDestroy{pascal_name}Descriptor(infiniop{pascal_name}Descriptor_t desc)"

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
