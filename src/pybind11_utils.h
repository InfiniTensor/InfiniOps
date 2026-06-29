#ifndef INFINI_OPS_PYBIND11_UTILS_H_
#define INFINI_OPS_PYBIND11_UTILS_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"
#include "torch/device_.h"

namespace py = pybind11;

namespace infini::ops {

namespace detail {

template <Device::Type... kDevs>
std::unordered_map<std::string, Device::Type> BuildTorchNameMap(
    List<kDevs...>) {
  std::unordered_map<std::string, Device::Type> map;
  (map.emplace(std::string{TorchDeviceName<kDevs>::kValue}, kDevs), ...);
  return map;
}

}  // namespace detail

inline DataType DataTypeFromString(const std::string& name) {
  return kStringToDataType.at(name);
}

inline std::string DataTypeToTorchName(DataType dtype) {
  switch (dtype) {
    case DataType::kInt8:
      return "int8";
    case DataType::kInt16:
      return "int16";
    case DataType::kInt32:
      return "int32";
    case DataType::kInt64:
      return "int64";
    case DataType::kUInt8:
      return "uint8";
    case DataType::kUInt16:
      return "uint16";
    case DataType::kUInt32:
      return "uint32";
    case DataType::kUInt64:
      return "uint64";
    case DataType::kFloat16:
      return "float16";
    case DataType::kBFloat16:
      return "bfloat16";
    case DataType::kFloat32:
      return "float32";
    case DataType::kFloat64:
      return "float64";
    default:
      throw py::value_error("Unsupported dtype for PyTorch tensor creation");
  }
}

inline py::object TorchDType(DataType dtype) {
  auto torch = py::module_::import("torch");
  auto name = DataTypeToTorchName(dtype);

  if (!py::hasattr(torch, name.c_str())) {
    throw py::value_error("Current PyTorch build does not expose dtype `" +
                          name + "`");
  }

  return torch.attr(name.c_str());
}

inline std::string TorchDeviceTypeName(Device::Type type) {
  switch (type) {
    case Device::Type::kCpu:
      return std::string{detail::TorchDeviceName<Device::Type::kCpu>::kValue};
    case Device::Type::kNvidia:
      return std::string{
          detail::TorchDeviceName<Device::Type::kNvidia>::kValue};
    case Device::Type::kCambricon:
      return std::string{
          detail::TorchDeviceName<Device::Type::kCambricon>::kValue};
    case Device::Type::kAscend:
      return std::string{
          detail::TorchDeviceName<Device::Type::kAscend>::kValue};
    case Device::Type::kMetax:
      return std::string{detail::TorchDeviceName<Device::Type::kMetax>::kValue};
    case Device::Type::kMoore:
      return std::string{detail::TorchDeviceName<Device::Type::kMoore>::kValue};
    case Device::Type::kIluvatar:
      return std::string{
          detail::TorchDeviceName<Device::Type::kIluvatar>::kValue};
    case Device::Type::kKunlun:
      return std::string{
          detail::TorchDeviceName<Device::Type::kKunlun>::kValue};
    case Device::Type::kHygon:
      return std::string{detail::TorchDeviceName<Device::Type::kHygon>::kValue};
    case Device::Type::kQy:
      return std::string{detail::TorchDeviceName<Device::Type::kQy>::kValue};
    default:
      throw py::value_error("Unsupported device for PyTorch tensor creation");
  }
}

inline py::object TorchDevice(Device device) {
  auto torch = py::module_::import("torch");
  auto name = TorchDeviceTypeName(device.type());

  if (device.type() != Device::Type::kCpu) {
    name += ":" + std::to_string(device.index());
  }

  return torch.attr("device")(name);
}

inline Tensor::Strides ContiguousStrides(const Tensor::Shape& shape) {
  Tensor::Strides strides(shape.size(), 1);

  for (std::size_t i = shape.size(); i > 1; --i) {
    strides[i - 2] = strides[i - 1] * shape[i - 1];
  }

  return strides;
}

template <typename T = void>
inline Device::Type DeviceTypeFromString(const std::string& name) {
  static const auto kTorchNameToTypes{
      detail::BuildTorchNameMap(ActiveDevices<T>{})};

  auto it{kTorchNameToTypes.find(name)};

  if (it != kTorchNameToTypes.cend()) {
    return it->second;
  }

  std::vector<std::string> supported_names;

  for (const auto& [torch_name, device_type] : kTorchNameToTypes) {
    const auto internal_name = std::string{Device::StringFromType(device_type)};

    if (name == internal_name) {
      return device_type;
    }

    supported_names.push_back(torch_name);
    supported_names.push_back(internal_name);
  }

  std::sort(supported_names.begin(), supported_names.end());
  supported_names.erase(
      std::unique(supported_names.begin(), supported_names.end()),
      supported_names.end());

  std::string message = "Unsupported device type `" + name +
                        "` for this InfiniOps build. Supported device names: ";

  for (std::size_t i = 0; i < supported_names.size(); ++i) {
    if (i != 0) {
      message += ", ";
    }
    message += supported_names[i];
  }

  message += ". Rebuild InfiniOps with the matching backend enabled.";

  throw py::value_error(message);
}

// Returns `nullopt` rather than aborting when the name does not resolve.
// Used by generated pybind bindings to query implementation indices for
// devices an op may not support, without crashing the process.
template <typename T = void>
inline std::optional<Device::Type> TryDeviceTypeFromString(
    const std::string& name) {
  static const auto kTorchNameToTypes{
      detail::BuildTorchNameMap(ActiveDevices<T>{})};

  auto it{kTorchNameToTypes.find(name)};

  if (it != kTorchNameToTypes.cend()) {
    return it->second;
  }

  static const std::unordered_map<std::string, Device::Type> kPlatformNames{
      {"cpu", Device::Type::kCpu},
      {"nvidia", Device::Type::kNvidia},
      {"cambricon", Device::Type::kCambricon},
      {"ascend", Device::Type::kAscend},
      {"metax", Device::Type::kMetax},
      {"moore", Device::Type::kMoore},
      {"iluvatar", Device::Type::kIluvatar},
      {"kunlun", Device::Type::kKunlun},
      {"hygon", Device::Type::kHygon},
      {"qy", Device::Type::kQy},
  };

  auto platform_it{kPlatformNames.find(name)};

  if (platform_it != kPlatformNames.cend()) {
    return platform_it->second;
  }

  return std::nullopt;
}

inline Device DeviceFromPybind11Handle(py::handle obj) {
  auto device_obj{obj.attr("device")};
  auto device_type_str{device_obj.attr("type").cast<std::string>()};
  auto device_index_obj{device_obj.attr("index")};
  auto device_index{device_index_obj.is_none() ? 0
                                               : device_index_obj.cast<int>()};

  return Device{DeviceTypeFromString(device_type_str), device_index};
}

inline Tensor TensorFromPybind11Handle(py::handle obj) {
  auto data{
      reinterpret_cast<void*>(obj.attr("data_ptr")().cast<std::uintptr_t>())};

  auto shape{obj.attr("shape").cast<typename Tensor::Shape>()};

  auto dtype_str{py::str(obj.attr("dtype")).cast<std::string>()};
  auto pos{dtype_str.find_last_of('.')};
  auto dtype{DataTypeFromString(
      pos == std::string::npos ? dtype_str : dtype_str.substr(pos + 1))};

  auto device{DeviceFromPybind11Handle(obj)};

  auto strides{obj.attr("stride")().cast<typename Tensor::Strides>()};

  return Tensor{data, std::move(shape), dtype, device, std::move(strides)};
}

inline std::optional<Tensor> OptionalTensorFromPybind11Handle(
    const std::optional<py::object>& obj) {
  if (!obj.has_value() || obj->is_none()) return std::nullopt;
  return TensorFromPybind11Handle(*obj);
}

inline std::vector<Tensor> VectorTensorFromPybind11Handle(
    const std::vector<py::object>& objs) {
  std::vector<Tensor> result;
  result.reserve(objs.size());
  for (const auto& obj : objs) {
    result.push_back(TensorFromPybind11Handle(obj));
  }
  return result;
}

class Pybind11Tensor {
 public:
  explicit Pybind11Tensor(py::handle obj)
      : object_{py::reinterpret_borrow<py::object>(obj)},
        shape_{object_.attr("shape").cast<Tensor::Shape>()},
        strides_{object_.attr("stride")().cast<Tensor::Strides>()},
        dtype_{TensorFromPybind11Handle(object_).dtype()},
        device_{DeviceFromPybind11Handle(object_)} {}

  static Pybind11Tensor Empty(const Tensor::Shape& shape, DataType dtype,
                              Device device) {
    auto torch = py::module_::import("torch");
    auto strides = ContiguousStrides(shape);
    auto object = torch.attr("empty_strided")(
        shape, strides, py::arg("dtype") = TorchDType(dtype),
        py::arg("device") = TorchDevice(device));

    return Pybind11Tensor{object};
  }

  void* data() const {
    return reinterpret_cast<void*>(
        object_.attr("data_ptr")().cast<std::uintptr_t>());
  }

  const Tensor::Shape& shape() const { return shape_; }

  const Tensor::Strides& strides() const { return strides_; }

  DataType dtype() const { return dtype_; }

  Device device() const { return device_; }

  const py::object& object() const { return object_; }

 private:
  py::object object_;

  Tensor::Shape shape_;

  Tensor::Strides strides_;

  DataType dtype_;

  Device device_;
};

}  // namespace infini::ops

#endif
