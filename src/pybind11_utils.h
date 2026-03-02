#ifndef INFINI_OPS_PYBIND11_UTILS_H_
#define INFINI_OPS_PYBIND11_UTILS_H_

#include <string>
#include <unordered_map>

#include "data_type.h"
#include "device.h"

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

}  // namespace infini::ops

#endif
