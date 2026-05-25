#include "infini/ops.h"

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <vector>

struct InfiniOpsHandlePrivate {
  infini::ops::Handle handle;
};

struct InfiniOpsConfigPrivate {
  infini::ops::Config config;
};

namespace {

thread_local std::string last_error;

void SetLastError(const char* message) { last_error = message; }

InfiniOpsStatus InvalidArgument(const char* message) {
  SetLastError(message);
  return INFINI_OPS_STATUS_INVALID_ARGUMENT;
}

bool IsValidDataType(InfiniOpsDataType data_type) {
  switch (data_type) {
    case INFINI_OPS_DATA_TYPE_FLOAT16:
    case INFINI_OPS_DATA_TYPE_BFLOAT16:
    case INFINI_OPS_DATA_TYPE_FLOAT32:
    case INFINI_OPS_DATA_TYPE_FLOAT64:
    case INFINI_OPS_DATA_TYPE_INT8:
    case INFINI_OPS_DATA_TYPE_INT16:
    case INFINI_OPS_DATA_TYPE_INT32:
    case INFINI_OPS_DATA_TYPE_INT64:
    case INFINI_OPS_DATA_TYPE_UINT8:
    case INFINI_OPS_DATA_TYPE_UINT16:
    case INFINI_OPS_DATA_TYPE_UINT32:
    case INFINI_OPS_DATA_TYPE_UINT64:
      return true;
    case INFINI_OPS_DATA_TYPE_INVALID:
      return false;
  }
  return false;
}

bool IsValidDeviceType(InfiniOpsDeviceType device_type) {
  switch (device_type) {
    case INFINI_OPS_DEVICE_TYPE_CPU:
    case INFINI_OPS_DEVICE_TYPE_NVIDIA:
    case INFINI_OPS_DEVICE_TYPE_CAMBRICON:
    case INFINI_OPS_DEVICE_TYPE_ASCEND:
    case INFINI_OPS_DEVICE_TYPE_METAX:
    case INFINI_OPS_DEVICE_TYPE_MOORE:
    case INFINI_OPS_DEVICE_TYPE_ILUVATAR:
      return true;
    case INFINI_OPS_DEVICE_TYPE_INVALID:
      return false;
  }
  return false;
}

bool ConvertDataType(InfiniOpsDataType input, infini::ops::DataType* output) {
  assert(output != nullptr);
  switch (input) {
    case INFINI_OPS_DATA_TYPE_FLOAT16:
      *output = infini::ops::DataType::kFloat16;
      return true;
    case INFINI_OPS_DATA_TYPE_FLOAT32:
      *output = infini::ops::DataType::kFloat32;
      return true;
    case INFINI_OPS_DATA_TYPE_FLOAT64:
      *output = infini::ops::DataType::kFloat64;
      return true;
    case INFINI_OPS_DATA_TYPE_BFLOAT16:
      *output = infini::ops::DataType::kBFloat16;
      return true;
    case INFINI_OPS_DATA_TYPE_INT8:
      *output = infini::ops::DataType::kInt8;
      return true;
    case INFINI_OPS_DATA_TYPE_INT16:
      *output = infini::ops::DataType::kInt16;
      return true;
    case INFINI_OPS_DATA_TYPE_INT32:
      *output = infini::ops::DataType::kInt32;
      return true;
    case INFINI_OPS_DATA_TYPE_INT64:
      *output = infini::ops::DataType::kInt64;
      return true;
    case INFINI_OPS_DATA_TYPE_UINT8:
      *output = infini::ops::DataType::kUInt8;
      return true;
    case INFINI_OPS_DATA_TYPE_UINT16:
      *output = infini::ops::DataType::kUInt16;
      return true;
    case INFINI_OPS_DATA_TYPE_UINT32:
      *output = infini::ops::DataType::kUInt32;
      return true;
    case INFINI_OPS_DATA_TYPE_UINT64:
      *output = infini::ops::DataType::kUInt64;
      return true;
    case INFINI_OPS_DATA_TYPE_INVALID:
      return false;
  }
  return false;
}

bool ConvertDeviceType(InfiniOpsDeviceType input,
                       infini::ops::Device::Type* output) {
  assert(output != nullptr);
  switch (input) {
    case INFINI_OPS_DEVICE_TYPE_CPU:
      *output = infini::ops::Device::Type::kCpu;
      return true;
    case INFINI_OPS_DEVICE_TYPE_NVIDIA:
      *output = infini::ops::Device::Type::kNvidia;
      return true;
    case INFINI_OPS_DEVICE_TYPE_CAMBRICON:
      *output = infini::ops::Device::Type::kCambricon;
      return true;
    case INFINI_OPS_DEVICE_TYPE_ASCEND:
      *output = infini::ops::Device::Type::kAscend;
      return true;
    case INFINI_OPS_DEVICE_TYPE_METAX:
      *output = infini::ops::Device::Type::kMetax;
      return true;
    case INFINI_OPS_DEVICE_TYPE_MOORE:
      *output = infini::ops::Device::Type::kMoore;
      return true;
    case INFINI_OPS_DEVICE_TYPE_ILUVATAR:
      *output = infini::ops::Device::Type::kIluvatar;
      return true;
    case INFINI_OPS_DEVICE_TYPE_INVALID:
      return false;
  }
  return false;
}

bool DataTypeSize(InfiniOpsDataType data_type, size_t* size) {
  assert(size != nullptr);
  switch (data_type) {
    case INFINI_OPS_DATA_TYPE_FLOAT16:
    case INFINI_OPS_DATA_TYPE_BFLOAT16:
    case INFINI_OPS_DATA_TYPE_INT16:
    case INFINI_OPS_DATA_TYPE_UINT16:
      *size = 2;
      return true;
    case INFINI_OPS_DATA_TYPE_FLOAT32:
    case INFINI_OPS_DATA_TYPE_INT32:
    case INFINI_OPS_DATA_TYPE_UINT32:
      *size = 4;
      return true;
    case INFINI_OPS_DATA_TYPE_FLOAT64:
    case INFINI_OPS_DATA_TYPE_INT64:
    case INFINI_OPS_DATA_TYPE_UINT64:
      *size = 8;
      return true;
    case INFINI_OPS_DATA_TYPE_INT8:
    case INFINI_OPS_DATA_TYPE_UINT8:
      *size = 1;
      return true;
    case INFINI_OPS_DATA_TYPE_INVALID:
      return false;
  }
  return false;
}

bool CheckedMultiply(size_t left, size_t right, size_t* result) {
  assert(result != nullptr);
  if (right != 0 && left > SIZE_MAX / right) {
    return false;
  }
  *result = left * right;
  return true;
}

bool ComputeRequiredByteSize(const InfiniOpsTensor& tensor,
                             size_t* required_byte_size) {
  assert(required_byte_size != nullptr);

  size_t element_size = 0;
  if (!DataTypeSize(tensor.data_type, &element_size)) {
    return false;
  }

  if (tensor.rank == 0) {
    *required_byte_size = element_size;
    return true;
  }

  size_t element_count = 1;
  if (tensor.stride == nullptr) {
    for (int32_t i = 0; i < tensor.rank; ++i) {
      if (!CheckedMultiply(element_count, static_cast<size_t>(tensor.shape[i]),
                           &element_count)) {
        return false;
      }
    }
    return CheckedMultiply(element_count, element_size, required_byte_size);
  }

  size_t max_offset = 0;
  for (int32_t i = 0; i < tensor.rank; ++i) {
    if (tensor.shape[i] == 0) {
      *required_byte_size = 0;
      return true;
    }
    if (tensor.stride[i] < 0) {
      return false;
    }
    size_t dimension_extent = 0;
    if (!CheckedMultiply(static_cast<size_t>(tensor.shape[i] - 1),
                         static_cast<size_t>(tensor.stride[i]),
                         &dimension_extent)) {
      return false;
    }
    if (SIZE_MAX - max_offset < dimension_extent) {
      return false;
    }
    max_offset += dimension_extent;
  }

  if (max_offset == SIZE_MAX) {
    return false;
  }
  return CheckedMultiply(max_offset + 1, element_size, required_byte_size);
}

std::vector<infini::ops::Tensor::Size> ConvertShape(
    const InfiniOpsTensor& tensor) {
  std::vector<infini::ops::Tensor::Size> result;
  result.reserve(tensor.rank);
  for (int32_t i = 0; i < tensor.rank; ++i) {
    result.push_back(static_cast<infini::ops::Tensor::Size>(tensor.shape[i]));
  }
  return result;
}

std::vector<infini::ops::Tensor::Stride> ConvertStrides(
    const InfiniOpsTensor& tensor) {
  std::vector<infini::ops::Tensor::Stride> result;
  result.reserve(tensor.rank);
  if (tensor.stride == nullptr) {
    return result;
  }
  for (int32_t i = 0; i < tensor.rank; ++i) {
    result.push_back(
        static_cast<infini::ops::Tensor::Stride>(tensor.stride[i]));
  }
  return result;
}

infini::ops::Tensor ToInternalTensor(const InfiniOpsTensor& tensor) {
  infini::ops::DataType data_type;
  const bool data_type_valid = ConvertDataType(tensor.data_type, &data_type);
  assert(data_type_valid);

  infini::ops::Device::Type device_type;
  const bool device_type_valid =
      ConvertDeviceType(tensor.device_type, &device_type);
  assert(device_type_valid);

  const auto shape = ConvertShape(tensor);
  const infini::ops::Device device(device_type);
  if (tensor.stride == nullptr) {
    return infini::ops::Tensor(tensor.data, shape, data_type, device);
  }
  return infini::ops::Tensor(tensor.data, shape, data_type, device,
                             ConvertStrides(tensor));
}

InfiniOpsStatus ValidateTensor(const char* name,
                               const InfiniOpsTensor* tensor) {
  if (tensor == nullptr) {
    SetLastError(name);
    last_error += " tensor must not be null";
    return INFINI_OPS_STATUS_INVALID_ARGUMENT;
  }
  if (tensor->structure_size < offsetof(InfiniOpsTensor, reserved)) {
    SetLastError(name);
    last_error += " tensor structure size is invalid";
    return INFINI_OPS_STATUS_INVALID_ARGUMENT;
  }
  if (!IsValidDataType(tensor->data_type)) {
    SetLastError(name);
    last_error += " tensor data type is invalid";
    return INFINI_OPS_STATUS_INVALID_ARGUMENT;
  }
  if (!IsValidDeviceType(tensor->device_type)) {
    SetLastError(name);
    last_error += " tensor device type is invalid";
    return INFINI_OPS_STATUS_INVALID_ARGUMENT;
  }
  if (tensor->rank < 0) {
    SetLastError(name);
    last_error += " tensor rank must not be negative";
    return INFINI_OPS_STATUS_INVALID_ARGUMENT;
  }
  if (tensor->rank > 0 && tensor->shape == nullptr) {
    SetLastError(name);
    last_error += " tensor shape must not be null for non-scalar";
    return INFINI_OPS_STATUS_INVALID_ARGUMENT;
  }
  for (int32_t i = 0; i < tensor->rank; ++i) {
    if (tensor->shape[i] < 0) {
      SetLastError(name);
      last_error += " tensor shape must not contain negative values";
      return INFINI_OPS_STATUS_INVALID_ARGUMENT;
    }
  }

  size_t required_byte_size = 0;
  if (!ComputeRequiredByteSize(*tensor, &required_byte_size)) {
    SetLastError(name);
    last_error += " tensor byte size is invalid";
    return INFINI_OPS_STATUS_INVALID_ARGUMENT;
  }
  if (required_byte_size > 0 && tensor->data == nullptr) {
    SetLastError(name);
    last_error += " tensor data must not be null";
    return INFINI_OPS_STATUS_INVALID_ARGUMENT;
  }
  if (tensor->byte_size < required_byte_size) {
    SetLastError(name);
    last_error += " tensor byte size is smaller than shape requires";
    return INFINI_OPS_STATUS_INVALID_ARGUMENT;
  }
  return INFINI_OPS_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

InfiniOpsStatus infiniOpsGetLastError(char* buffer, size_t capacity,
                                      size_t* required_size) {
  const size_t required = last_error.size() + 1;
  if (required_size != nullptr) {
    *required_size = required;
  }
  if (buffer == nullptr || capacity == 0) {
    return last_error.empty() ? INFINI_OPS_STATUS_SUCCESS
                              : INFINI_OPS_STATUS_INVALID_ARGUMENT;
  }
  if (capacity < required) {
    if (capacity > 0) {
      buffer[0] = '\0';
    }
    return INFINI_OPS_STATUS_INVALID_ARGUMENT;
  }
  std::memcpy(buffer, last_error.c_str(), required);
  return INFINI_OPS_STATUS_SUCCESS;
}

InfiniOpsStatus infiniOpsCreateHandle(
    const InfiniOpsHandleAttributes* attributes, InfiniOpsHandle* handle) {
  try {
    if (handle == nullptr) {
      return InvalidArgument("handle output must not be null");
    }
    *handle = nullptr;
    if (attributes == nullptr) {
      return InvalidArgument("handle attributes must not be null");
    }
    if (attributes->structure_size <
        offsetof(InfiniOpsHandleAttributes, reserved)) {
      return InvalidArgument("handle attributes size is invalid");
    }
    if (attributes->workspace_byte_size > 0 &&
        attributes->workspace == nullptr) {
      return InvalidArgument("handle workspace must not be null");
    }

    InfiniOpsHandlePrivate* created = new InfiniOpsHandlePrivate;
    created->handle.set_stream(reinterpret_cast<void*>(attributes->stream));
    created->handle.set_workspace(attributes->workspace);
    created->handle.set_workspace_size_in_bytes(
        attributes->workspace_byte_size);
    *handle = created;
    SetLastError("");
    return INFINI_OPS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    SetLastError("out of memory while creating handle");
    return INFINI_OPS_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    SetLastError(error.what());
    return INFINI_OPS_STATUS_INTERNAL_ERROR;
  } catch (...) {
    SetLastError("unknown error while creating handle");
    return INFINI_OPS_STATUS_INTERNAL_ERROR;
  }
}

InfiniOpsStatus infiniOpsDestroyHandle(InfiniOpsHandle handle) {
  delete handle;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

InfiniOpsStatus infiniOpsCreateConfig(
    const InfiniOpsConfigAttributes* attributes, InfiniOpsConfig* config) {
  try {
    if (config == nullptr) {
      return InvalidArgument("config output must not be null");
    }
    *config = nullptr;
    if (attributes == nullptr) {
      return InvalidArgument("config attributes must not be null");
    }
    if (attributes->structure_size <
        offsetof(InfiniOpsConfigAttributes, reserved)) {
      return InvalidArgument("config attributes size is invalid");
    }

    InfiniOpsConfigPrivate* created = new InfiniOpsConfigPrivate;
    created->config.set_implementation_index(attributes->implementation_index);
    *config = created;
    SetLastError("");
    return INFINI_OPS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    SetLastError("out of memory while creating config");
    return INFINI_OPS_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    SetLastError(error.what());
    return INFINI_OPS_STATUS_INTERNAL_ERROR;
  } catch (...) {
    SetLastError("unknown error while creating config");
    return INFINI_OPS_STATUS_INTERNAL_ERROR;
  }
}

InfiniOpsStatus infiniOpsDestroyConfig(InfiniOpsConfig config) {
  delete config;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

#include "c_ops.inc"

}  // extern "C"
