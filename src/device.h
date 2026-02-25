#ifndef INFINI_OPS_DEVICE_H_
#define INFINI_OPS_DEVICE_H_

#include "common/constexpr_map.h"
#include "common/traits.h"

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
  template <Device::Type dev,
            std::enable_if_t<dev == Device::Type::kCpu, int> = 0>
  void operator()() const {}
#endif

#ifdef WITH_NVIDIA
  template <Device::Type dev,
            std::enable_if_t<dev == Device::Type::kNvidia, int> = 0>
  void operator()() const {}
#endif

#ifdef WITH_CAMBRICON
  template <Device::Type dev,
            std::enable_if_t<dev == Device::Type::kCambricon, int> = 0>
  void operator()() const {}
#endif

#ifdef WITH_ASCEND
  template <Device::Type dev,
            std::enable_if_t<dev == Device::Type::kAscend, int> = 0>
  void operator()() const {}
#endif

#ifdef WITH_METAX
  template <Device::Type dev,
            std::enable_if_t<dev == Device::Type::kMetax, int> = 0>
  void operator()() const {}
#endif

#ifdef WITH_MOORE
  template <Device::Type dev,
            std::enable_if_t<dev == Device::Type::kMoore, int> = 0>
  void operator()() const {}
#endif

#ifdef WITH_ILUVATAR
  template <Device::Type dev,
            std::enable_if_t<dev == Device::Type::kIluvatar, int> = 0>
  void operator()() const {}
#endif

#ifdef WITH_KUNLUN
  template <Device::Type dev,
            std::enable_if_t<dev == Device::Type::kKunlun, int> = 0>
  void operator()() const {}
#endif

#ifdef WITH_HYGON
  template <Device::Type dev,
            std::enable_if_t<dev == Device::Type::kHygon, int> = 0>
  void operator()() const {}
#endif

#ifdef WITH_QY
  template <Device::Type dev,
            std::enable_if_t<dev == Device::Type::kQy, int> = 0>
  void operator()() const {}
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

#endif
