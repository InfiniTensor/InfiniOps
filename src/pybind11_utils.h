#ifndef INFINI_OPS_PYBIND11_UTILS_H_
#define INFINI_OPS_PYBIND11_UTILS_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

  return Device::TypeFromString(name);
}

inline Tensor TensorFromPybind11Handle(py::handle obj) {
  auto data{
      reinterpret_cast<void*>(obj.attr("data_ptr")().cast<std::uintptr_t>())};

  // pybind11 converts Python sequences to `std::vector` via `stl.h`; our
  // `SmallVector` isn't in that map, so iterate manually.
  Tensor::Shape shape;
  for (auto s : obj.attr("shape")) {
    shape.push_back(s.cast<Tensor::Size>());
  }

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

  Tensor::Strides strides;
  for (auto s : obj.attr("stride")()) {
    strides.push_back(s.cast<Tensor::Stride>());
  }

  return Tensor{data, std::move(shape), dtype, device, std::move(strides)};
}

inline std::optional<Tensor> OptionalTensorFromPybind11Handle(
    const std::optional<py::object>& obj) {
  if (!obj.has_value()) return std::nullopt;
  return TensorFromPybind11Handle(*obj);
}

}  // namespace infini::ops

#endif
