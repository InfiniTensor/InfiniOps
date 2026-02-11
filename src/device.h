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

 private:
  Type type_{Type::kCpu};

  static constexpr ConstexprMap<Device::Type, std::string_view, 10>
      kDeviceToDesc{{{
          {Device::Type::kCpu, "CPU"},
          {Device::Type::kNvidia, "NVIDIA"},
          {Device::Type::kCambricon, "Cambricon"},
          {Device::Type::kAscend, "Ascend"},
          {Device::Type::kMetax, "Metax"},
          {Device::Type::kMoore, "Moore"},
          {Device::Type::kIluvatar, "Iluvatar"},
          {Device::Type::kKunlun, "Kunlun"},
          {Device::Type::kHygon, "Hygon"},
          {Device::Type::kQy, "QY"},
      }}};

  static constexpr ConstexprMap<std::string_view, Device::Type, 10>
      kDescToDevice{{{
          {"CPU", Device::Type::kCpu},
          {"NVIDIA", Device::Type::kNvidia},
          {"Cambricon", Device::Type::kCambricon},
          {"Ascend", Device::Type::kAscend},
          {"Metax", Device::Type::kMetax},
          {"Moore", Device::Type::kMoore},
          {"Iluvatar", Device::Type::kIluvatar},
          {"Kunlun", Device::Type::kKunlun},
          {"Hygon", Device::Type::kHygon},
          {"QY", Device::Type::kQy},
      }}};

  int index_{0};
};

struct EnabledDeviceFilter {
  // Each block defines a template operator() specialized for a specific
  // Device. If the macro is NOT defined, the specialization is not compiled,
  // and FilterList will exclude it from ActiveDevices.

#ifdef USE_CPU
  template <Device::Type D, std::enable_if_t<D == Device::Type::kCpu, int> = 0>
  void operator()() const {}
#endif

#ifdef USE_NVIDIA
  template <Device::Type D,
            std::enable_if_t<D == Device::Type::kNvidia, int> = 0>
  void operator()() const {}
#endif

#ifdef USE_CAMBRICON
  template <Device::Type D,
            std::enable_if_t<D == Device::Type::kCambricon, int> = 0>
  void operator()() const {}
#endif

#ifdef USE_ASCEND
  template <Device::Type D,
            std::enable_if_t<D == Device::Type::kAscend, int> = 0>
  void operator()() const {}
#endif

#ifdef USE_METAX
  template <Device::Type D,
            std::enable_if_t<D == Device::Type::kMetax, int> = 0>
  void operator()() const {}
#endif

#ifdef USE_MOORE
  template <Device::Type D,
            std::enable_if_t<D == Device::Type::kMoore, int> = 0>
  void operator()() const {}
#endif

#ifdef USE_ILUVATAR
  template <Device::Type D,
            std::enable_if_t<D == Device::Type::kIluvatar, int> = 0>
  void operator()() const {}
#endif

#ifdef USE_KUNLUN
  template <Device::Type D,
            std::enable_if_t<D == Device::Type::kKunlun, int> = 0>
  void operator()() const {}
#endif

#ifdef USE_HYGON
  template <Device::Type D,
            std::enable_if_t<D == Device::Type::kHygon, int> = 0>
  void operator()() const {}
#endif

#ifdef USE_QY
  template <Device::Type D, std::enable_if_t<D == Device::Type::kQy, int> = 0>
  void operator()() const {}
#endif
};

// Defines the common categories of devices using List
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
