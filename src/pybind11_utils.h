#ifndef INFINI_OPS_PYBIND11_UTILS_H_
#define INFINI_OPS_PYBIND11_UTILS_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>

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

inline Tensor TensorFromPybind11Handle(py::handle obj) {
  auto data{
      reinterpret_cast<void*>(obj.attr("data_ptr")().cast<std::uintptr_t>())};

  auto shape{obj.attr("shape").cast<typename Tensor::Shape>()};

  auto dtype_str{py::str(obj.attr("dtype")).cast<std::string>()};
  auto pos{dtype_str.find_last_of('.')};
  auto dtype{DataTypeFromString(
      pos == std::string::npos ? dtype_str : dtype_str.substr(pos + 1))};

  auto device_obj{obj.attr("device")};
  auto device_type_str{device_obj.attr("type").cast<std::string>()};
  auto device_index_obj{device_obj.attr("index")};
  auto device_index{device_index_obj.is_none() ? 0
                                               : device_index_obj.cast<int>()};
  Device device{DeviceTypeFromString(device_type_str), device_index};

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

}  // namespace infini::ops

#endif
