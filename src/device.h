#ifndef INFINI_OPS_DEVICE_H_
#define INFINI_OPS_DEVICE_H_

#include "common/constexpr_map.h"
#include "common/traits.h"
#include "hash.h"

namespace infini::ops {

class Device {
 public:
  enum class Type {
    kCpu = 0,
    kNvidia = 1,
    kCambricon = 2,
    kAscend = 3,
    kMetax = 4,
    kMoore = 5,
    kIluvatar = 6,
    kKunlun = 7,
    kHygon = 8,
    kQy = 9,
    kCount
  };

  Device() = default;

  Device(const Type& type, const int& index = 0) : type_{type}, index_{index} {}

  static const Type TypeFromString(const std::string& name) {
    return kDescToDevice.at(name);
  }

  static const std::string_view StringFromType(const Type& type) {
    return kDeviceToDesc.at(type);
  }

  const Type& type() const { return type_; }

  const int& index() const { return index_; }

  std::string ToString() const {
    return std::string{StringFromType(type_)} + ":" + std::to_string(index_);
  }

 private:
  Type type_{Type::kCpu};

  static constexpr ConstexprMap<Device::Type, std::string_view,
                                static_cast<std::size_t>(Device::Type::kCount)>
      kDeviceToDesc{{{
          {Type::kCpu, "cpu"},
          {Type::kNvidia, "nvidia"},
          {Type::kCambricon, "cambricon"},
          {Type::kAscend, "ascend"},
          {Type::kMetax, "metax"},
          {Type::kMoore, "moore"},
          {Type::kIluvatar, "iluvatar"},
          {Type::kKunlun, "kunlun"},
          {Type::kHygon, "hygon"},
          {Type::kQy, "qy"},
      }}};

  static constexpr ConstexprMap<std::string_view, Device::Type,
                                static_cast<std::size_t>(Device::Type::kCount)>
      kDescToDevice{{{
          {"cpu", Type::kCpu},
          {"nvidia", Type::kNvidia},
          {"cambricon", Type::kCambricon},
          {"ascend", Type::kAscend},
          {"metax", Type::kMetax},
          {"moore", Type::kMoore},
          {"iluvatar", Type::kIluvatar},
          {"kunlun", Type::kKunlun},
          {"hygon", Type::kHygon},
          {"qy", Type::kQy},
      }}};

  int index_{0};
};

struct EnabledDeviceFilter {
  // Each block defines a template operator() specialized for a specific
  // Device. If the macro is NOT defined, the specialization is not compiled,
  // and FilterList will exclude it from ActiveDevices.

#ifdef WITH_CPU
  void operator()(ValueTag<Device::Type::kCpu>) const {}
#endif

#ifdef WITH_NVIDIA
  void operator()(ValueTag<Device::Type::kNvidia>) const {}
#endif

#ifdef WITH_CAMBRICON
  void operator()(ValueTag<Device::Type::kCambricon>) const {}
#endif

#ifdef WITH_ASCEND
  void operator()(ValueTag<Device::Type::kAscend>) const {}
#endif

#ifdef WITH_METAX
  void operator()(ValueTag<Device::Type::kMetax>) const {}
#endif

#ifdef WITH_MOORE
  void operator()(ValueTag<Device::Type::kMoore>) const {}
#endif

#ifdef WITH_ILUVATAR
  void operator()(ValueTag<Device::Type::kIluvatar>) const {}
#endif

#ifdef WITH_KUNLUN
  void operator()(ValueTag<Device::Type::kKunlun>) const {}
#endif

#ifdef WITH_HYGON
  void operator()(ValueTag<Device::Type::kHygon>) const {}
#endif

#ifdef WITH_QY
  void operator()(ValueTag<Device::Type::kQy>) const {}
#endif
};

// Defines the common categories of devices using List.
using AllDeviceTypes =
    List<Device::Type::kCpu, Device::Type::kNvidia, Device::Type::kCambricon,
         Device::Type::kAscend, Device::Type::kMetax, Device::Type::kMoore,
         Device::Type::kIluvatar, Device::Type::kKunlun, Device::Type::kHygon,
         Device::Type::kQy>;

using ActiveDevices =
    typename infini::ops::FilterList<EnabledDeviceFilter, std::tuple<>,
                                     AllDeviceTypes>::type;

}  // namespace infini::ops

template <>
struct std::hash<infini::ops::Device> {
  std::size_t operator()(const infini::ops::Device& device) const {
    std::size_t seed{0};

    hash_combine(seed, device.type());

    hash_combine(seed, device.index());

    return seed;
  }
};

#endif
