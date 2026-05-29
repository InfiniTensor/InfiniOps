#include "infini/ops.h"

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <utility>
#include <vector>

struct InfiniOpsTensorPrivate {
  void* data = nullptr;
  size_t byte_size = 0;
  InfiniOpsDataType data_type = INFINI_OPS_DATA_TYPE_INVALID;
  InfiniOpsDeviceType device_type = INFINI_OPS_DEVICE_TYPE_INVALID;
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  bool has_stride = false;
};

struct InfiniOpsHandlePrivate {
  infini::ops::Handle handle;
  InfiniOpsStream stream = nullptr;
  void* workspace = nullptr;
  size_t workspace_byte_size = 0;
};

struct InfiniOpsConfigPrivate {
  infini::ops::Config config;
  size_t implementation_index = 0;
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

bool ComputeRequiredByteSize(const InfiniOpsTensorPrivate& tensor,
                             size_t* required_byte_size) {
  assert(required_byte_size != nullptr);

  size_t element_size = 0;
  if (!DataTypeSize(tensor.data_type, &element_size)) {
    return false;
  }

  if (tensor.shape.empty()) {
    *required_byte_size = element_size;
    return true;
  }

  size_t element_count = 1;
  if (!tensor.has_stride) {
    for (int64_t dimension : tensor.shape) {
      if (!CheckedMultiply(element_count, static_cast<size_t>(dimension),
                           &element_count)) {
        return false;
      }
    }
    return CheckedMultiply(element_count, element_size, required_byte_size);
  }

  size_t max_offset = 0;
  for (size_t i = 0; i < tensor.shape.size(); ++i) {
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
    const InfiniOpsTensorPrivate& tensor) {
  std::vector<infini::ops::Tensor::Size> result;
  result.reserve(tensor.shape.size());
  for (int64_t dimension : tensor.shape) {
    result.push_back(static_cast<infini::ops::Tensor::Size>(dimension));
  }
  return result;
}

std::vector<infini::ops::Tensor::Stride> ConvertStrides(
    const InfiniOpsTensorPrivate& tensor) {
  std::vector<infini::ops::Tensor::Stride> result;
  result.reserve(tensor.stride.size());
  if (!tensor.has_stride) {
    return result;
  }
  for (int64_t stride : tensor.stride) {
    result.push_back(static_cast<infini::ops::Tensor::Stride>(stride));
  }
  return result;
}

infini::ops::Tensor ToInternalTensor(const InfiniOpsTensorPrivate& tensor) {
  infini::ops::DataType data_type;
  const bool data_type_valid = ConvertDataType(tensor.data_type, &data_type);
  assert(data_type_valid);

  infini::ops::Device::Type device_type;
  const bool device_type_valid =
      ConvertDeviceType(tensor.device_type, &device_type);
  assert(device_type_valid);

  const auto shape = ConvertShape(tensor);
  const infini::ops::Device device(device_type);
  if (!tensor.has_stride) {
    return infini::ops::Tensor(tensor.data, shape, data_type, device);
  }
  return infini::ops::Tensor(tensor.data, shape, data_type, device,
                             ConvertStrides(tensor));
}

InfiniOpsStatus ValidateTensor(const char* name, InfiniOpsTensor tensor) {
  if (tensor == nullptr) {
    SetLastError(name);
    last_error += " tensor must not be null";
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
  for (int64_t dimension : tensor->shape) {
    if (dimension < 0) {
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

INFINI_OPS_API InfiniOpsStatus infiniOpsGetLastError(char* buffer,
                                                     size_t capacity,
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

INFINI_OPS_API InfiniOpsStatus infiniOpsCreateTensor(InfiniOpsTensor* tensor) {
  try {
    if (tensor == nullptr) {
      return InvalidArgument("tensor output must not be null");
    }
    *tensor = nullptr;
    *tensor = new InfiniOpsTensorPrivate;
    SetLastError("");
    return INFINI_OPS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    SetLastError("out of memory while creating tensor");
    return INFINI_OPS_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    SetLastError(error.what());
    return INFINI_OPS_STATUS_INTERNAL_ERROR;
  } catch (...) {
    SetLastError("unknown error while creating tensor");
    return INFINI_OPS_STATUS_INTERNAL_ERROR;
  }
}

INFINI_OPS_API InfiniOpsStatus infiniOpsDestroyTensor(InfiniOpsTensor tensor) {
  delete tensor;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus infiniOpsSetTensorData(InfiniOpsTensor tensor,
                                                      void* data) {
  if (tensor == nullptr) {
    return InvalidArgument("tensor must not be null");
  }
  tensor->data = data;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus infiniOpsGetTensorData(InfiniOpsTensor tensor,
                                                      void** data) {
  if (tensor == nullptr) {
    return InvalidArgument("tensor must not be null");
  }
  if (data == nullptr) {
    return InvalidArgument("tensor data output must not be null");
  }
  *data = tensor->data;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus
infiniOpsSetTensorByteSize(InfiniOpsTensor tensor, size_t byte_size) {
  if (tensor == nullptr) {
    return InvalidArgument("tensor must not be null");
  }
  tensor->byte_size = byte_size;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus
infiniOpsGetTensorByteSize(InfiniOpsTensor tensor, size_t* byte_size) {
  if (tensor == nullptr) {
    return InvalidArgument("tensor must not be null");
  }
  if (byte_size == nullptr) {
    return InvalidArgument("tensor byte size output must not be null");
  }
  *byte_size = tensor->byte_size;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus infiniOpsSetTensorDataType(
    InfiniOpsTensor tensor, InfiniOpsDataType data_type) {
  if (tensor == nullptr) {
    return InvalidArgument("tensor must not be null");
  }
  if (!IsValidDataType(data_type)) {
    return InvalidArgument("tensor data type is invalid");
  }
  tensor->data_type = data_type;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus infiniOpsGetTensorDataType(
    InfiniOpsTensor tensor, InfiniOpsDataType* data_type) {
  if (tensor == nullptr) {
    return InvalidArgument("tensor must not be null");
  }
  if (data_type == nullptr) {
    return InvalidArgument("tensor data type output must not be null");
  }
  *data_type = tensor->data_type;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus infiniOpsSetTensorDeviceType(
    InfiniOpsTensor tensor, InfiniOpsDeviceType device_type) {
  if (tensor == nullptr) {
    return InvalidArgument("tensor must not be null");
  }
  if (!IsValidDeviceType(device_type)) {
    return InvalidArgument("tensor device type is invalid");
  }
  tensor->device_type = device_type;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus infiniOpsGetTensorDeviceType(
    InfiniOpsTensor tensor, InfiniOpsDeviceType* device_type) {
  if (tensor == nullptr) {
    return InvalidArgument("tensor must not be null");
  }
  if (device_type == nullptr) {
    return InvalidArgument("tensor device type output must not be null");
  }
  *device_type = tensor->device_type;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus infiniOpsSetTensorShape(InfiniOpsTensor tensor,
                                                       int32_t rank,
                                                       const int64_t* shape) {
  try {
    if (tensor == nullptr) {
      return InvalidArgument("tensor must not be null");
    }
    if (rank < 0) {
      return InvalidArgument("tensor rank must not be negative");
    }
    if (rank > 0 && shape == nullptr) {
      return InvalidArgument("tensor shape must not be null for non-scalar");
    }
    std::vector<int64_t> new_shape;
    new_shape.reserve(static_cast<size_t>(rank));
    for (int32_t i = 0; i < rank; ++i) {
      if (shape[i] < 0) {
        return InvalidArgument("tensor shape must not contain negative values");
      }
      new_shape.push_back(shape[i]);
    }
    tensor->shape = std::move(new_shape);
    tensor->stride.clear();
    tensor->has_stride = false;
    SetLastError("");
    return INFINI_OPS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    SetLastError("out of memory while setting tensor shape");
    return INFINI_OPS_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    SetLastError(error.what());
    return INFINI_OPS_STATUS_INTERNAL_ERROR;
  } catch (...) {
    SetLastError("unknown error while setting tensor shape");
    return INFINI_OPS_STATUS_INTERNAL_ERROR;
  }
}

INFINI_OPS_API InfiniOpsStatus infiniOpsGetTensorShape(InfiniOpsTensor tensor,
                                                       int32_t* rank,
                                                       const int64_t** shape) {
  if (tensor == nullptr) {
    return InvalidArgument("tensor must not be null");
  }
  if (rank == nullptr) {
    return InvalidArgument("tensor rank output must not be null");
  }
  if (shape == nullptr) {
    return InvalidArgument("tensor shape output must not be null");
  }
  *rank = static_cast<int32_t>(tensor->shape.size());
  *shape = tensor->shape.empty() ? nullptr : tensor->shape.data();
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus infiniOpsSetTensorStride(InfiniOpsTensor tensor,
                                                        const int64_t* stride) {
  try {
    if (tensor == nullptr) {
      return InvalidArgument("tensor must not be null");
    }
    if (tensor->shape.empty()) {
      tensor->stride.clear();
      tensor->has_stride = false;
      SetLastError("");
      return INFINI_OPS_STATUS_SUCCESS;
    }
    if (stride == nullptr) {
      return InvalidArgument("tensor stride must not be null for non-scalar");
    }
    std::vector<int64_t> new_stride;
    new_stride.reserve(tensor->shape.size());
    for (size_t i = 0; i < tensor->shape.size(); ++i) {
      if (stride[i] < 0) {
        return InvalidArgument(
            "tensor stride must not contain negative values");
      }
      new_stride.push_back(stride[i]);
    }
    tensor->stride = std::move(new_stride);
    tensor->has_stride = true;
    SetLastError("");
    return INFINI_OPS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    SetLastError("out of memory while setting tensor stride");
    return INFINI_OPS_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    SetLastError(error.what());
    return INFINI_OPS_STATUS_INTERNAL_ERROR;
  } catch (...) {
    SetLastError("unknown error while setting tensor stride");
    return INFINI_OPS_STATUS_INTERNAL_ERROR;
  }
}

INFINI_OPS_API InfiniOpsStatus
infiniOpsClearTensorStride(InfiniOpsTensor tensor) {
  if (tensor == nullptr) {
    return InvalidArgument("tensor must not be null");
  }
  tensor->stride.clear();
  tensor->has_stride = false;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus
infiniOpsGetTensorStride(InfiniOpsTensor tensor, const int64_t** stride) {
  if (tensor == nullptr) {
    return InvalidArgument("tensor must not be null");
  }
  if (stride == nullptr) {
    return InvalidArgument("tensor stride output must not be null");
  }
  *stride = tensor->has_stride && !tensor->stride.empty()
                ? tensor->stride.data()
                : nullptr;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus infiniOpsCreateHandle(InfiniOpsHandle* handle) {
  try {
    if (handle == nullptr) {
      return InvalidArgument("handle output must not be null");
    }
    *handle = nullptr;
    *handle = new InfiniOpsHandlePrivate;
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

INFINI_OPS_API InfiniOpsStatus infiniOpsDestroyHandle(InfiniOpsHandle handle) {
  delete handle;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus
infiniOpsSetHandleStream(InfiniOpsHandle handle, InfiniOpsStream stream) {
  if (handle == nullptr) {
    return InvalidArgument("handle must not be null");
  }
  handle->stream = stream;
  handle->handle.set_stream(reinterpret_cast<void*>(stream));
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus
infiniOpsGetHandleStream(InfiniOpsHandle handle, InfiniOpsStream* stream) {
  if (handle == nullptr) {
    return InvalidArgument("handle must not be null");
  }
  if (stream == nullptr) {
    return InvalidArgument("handle stream output must not be null");
  }
  *stream = handle->stream;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus infiniOpsSetHandleWorkspace(
    InfiniOpsHandle handle, void* workspace, size_t byte_size) {
  if (handle == nullptr) {
    return InvalidArgument("handle must not be null");
  }
  if (byte_size > 0 && workspace == nullptr) {
    return InvalidArgument("handle workspace must not be null");
  }
  handle->workspace = workspace;
  handle->workspace_byte_size = byte_size;
  handle->handle.set_workspace(workspace);
  handle->handle.set_workspace_size_in_bytes(byte_size);
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus infiniOpsGetHandleWorkspace(
    InfiniOpsHandle handle, void** workspace, size_t* byte_size) {
  if (handle == nullptr) {
    return InvalidArgument("handle must not be null");
  }
  if (workspace == nullptr) {
    return InvalidArgument("handle workspace output must not be null");
  }
  if (byte_size == nullptr) {
    return InvalidArgument(
        "handle workspace byte size output must not be null");
  }
  *workspace = handle->workspace;
  *byte_size = handle->workspace_byte_size;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus infiniOpsCreateConfig(InfiniOpsConfig* config) {
  try {
    if (config == nullptr) {
      return InvalidArgument("config output must not be null");
    }
    *config = nullptr;
    *config = new InfiniOpsConfigPrivate;
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

INFINI_OPS_API InfiniOpsStatus infiniOpsDestroyConfig(InfiniOpsConfig config) {
  delete config;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus infiniOpsSetConfigImplementationIndex(
    InfiniOpsConfig config, size_t implementation_index) {
  if (config == nullptr) {
    return InvalidArgument("config must not be null");
  }
  config->implementation_index = implementation_index;
  config->config.set_implementation_index(implementation_index);
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

INFINI_OPS_API InfiniOpsStatus infiniOpsGetConfigImplementationIndex(
    InfiniOpsConfig config, size_t* implementation_index) {
  if (config == nullptr) {
    return InvalidArgument("config must not be null");
  }
  if (implementation_index == nullptr) {
    return InvalidArgument(
        "config implementation index output must not be null");
  }
  *implementation_index = config->implementation_index;
  SetLastError("");
  return INFINI_OPS_STATUS_SUCCESS;
}

#include "c_ops.inc"

}  // extern "C"
