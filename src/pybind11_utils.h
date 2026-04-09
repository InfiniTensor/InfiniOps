#ifndef INFINI_OPS_PYBIND11_UTILS_H_
#define INFINI_OPS_PYBIND11_UTILS_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor.h"

namespace py = pybind11;

namespace infini::ops {

namespace detail {

template <Device::Type kDev>
struct TorchDeviceName;

template <>
struct TorchDeviceName<Device::Type::kCpu> {
  static constexpr std::string_view kValue{"cpu"};
};

template <>
struct TorchDeviceName<Device::Type::kNvidia> {
  static constexpr std::string_view kValue{"cuda"};
};

template <>
struct TorchDeviceName<Device::Type::kMetax> {
  static constexpr std::string_view kValue{"cuda"};
};

template <>
struct TorchDeviceName<Device::Type::kIluvatar> {
  static constexpr std::string_view kValue{"cuda"};
};

template <>
struct TorchDeviceName<Device::Type::kKunlun> {
  static constexpr std::string_view kValue{"cuda"};
};

template <>
struct TorchDeviceName<Device::Type::kHygon> {
  static constexpr std::string_view kValue{"cuda"};
};

template <>
struct TorchDeviceName<Device::Type::kQy> {
  static constexpr std::string_view kValue{"cuda"};
};

template <>
struct TorchDeviceName<Device::Type::kCambricon> {
  static constexpr std::string_view kValue{"mlu"};
};

template <>
struct TorchDeviceName<Device::Type::kAscend> {
  static constexpr std::string_view kValue{"npu"};
};

template <>
struct TorchDeviceName<Device::Type::kMoore> {
  static constexpr std::string_view kValue{"musa"};
};

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
