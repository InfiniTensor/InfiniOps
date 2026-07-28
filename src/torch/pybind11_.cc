#include "torch/pybind11_.h"

#include <torch/version.h>

#include <string>

#include "torch/csrc/autograd/python_variable.h"

namespace infini::ops {

namespace {

std::optional<DataType> TryDataTypeFromAten(at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::kChar:
      return DataType::kInt8;
    case at::kShort:
      return DataType::kInt16;
    case at::kInt:
      return DataType::kInt32;
    case at::kLong:
      return DataType::kInt64;
    case at::kByte:
      return DataType::kUInt8;
    case at::kBool:
      return DataType::kUInt8;
    case at::kHalf:
      return DataType::kFloat16;
    case at::kBFloat16:
      return DataType::kBFloat16;
    case at::kFloat:
      return DataType::kFloat32;
    case at::kDouble:
      return DataType::kFloat64;
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 4)
    case at::ScalarType::UInt16:
      return DataType::kUInt16;
    case at::ScalarType::UInt32:
      return DataType::kUInt32;
    case at::ScalarType::UInt64:
      return DataType::kUInt64;
#endif
    default:
      return std::nullopt;
  }
}

std::string DeviceTypeFromAten(const c10::Device& device) {
  if (device.type() == c10::kCPU) {
    return "cpu";
  }

  if (device.type() == c10::kCUDA) {
    return "cuda";
  }

  std::string name{device.str()};
  auto colon{name.find(':')};
  return colon == std::string::npos ? name : name.substr(0, colon);
}

}  // namespace

std::optional<AtenTensorMetadata> TryAtenTensorMetadataFromPyObject(
    void* py_object) {
  auto* object{reinterpret_cast<PyObject*>(py_object)};
  if (!THPVariable_Check(object)) {
    return std::nullopt;
  }

  const at::Tensor& tensor{THPVariable_Unpack(object)};
  const c10::Device device{tensor.device()};
  auto dtype{TryDataTypeFromAten(tensor.scalar_type())};

  if (!dtype.has_value()) {
    return std::nullopt;
  }

  return AtenTensorMetadata{
      tensor.data_ptr(),
      Tensor::Shape(tensor.sizes().begin(), tensor.sizes().end()),
      Tensor::Strides(tensor.strides().begin(), tensor.strides().end()),
      *dtype,
      DeviceTypeFromAten(device),
      device.has_index() ? device.index() : 0,
  };
}

}  // namespace infini::ops
