#ifndef INFINI_OPS_PYBIND11_UTILS_H_
#define INFINI_OPS_PYBIND11_UTILS_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor.h"

namespace py = pybind11;

namespace infini::ops {

inline DataType DataTypeFromString(const std::string& name) {
  return kStringToDataType.at(name);
}

inline Device::Type DeviceTypeFromString(const std::string& name) {
  static const std::unordered_map<std::string, Device::Type> kTorchNameToTypes{
      {"cpu", Device::Type::kCpu},
#ifdef WITH_NVIDIA
      {"cuda", Device::Type::kNvidia},
#endif
#ifdef WITH_METAX
      {"cuda", Device::Type::kMetax},
#endif
#ifdef WITH_ILUVATAR
      {"cuda", Device::Type::kIluvatar},
#endif
#ifdef WITH_KUNLUN
      {"cuda", Device::Type::kKunlun},
#endif
#ifdef WITH_HYGON
      {"cuda", Device::Type::kHygon},
#endif
#ifdef WITH_QY
      {"cuda", Device::Type::kQy},
#endif
      {"mlu", Device::Type::kCambricon}, {"npu", Device::Type::kAscend},
      {"musa", Device::Type::kMoore}};

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

}  // namespace infini::ops

#endif
