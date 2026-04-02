#ifndef INFINI_OPS_CPU_RUNTIME_H_
#define INFINI_OPS_CPU_RUNTIME_H_

#include "runtime.h"

namespace infini::ops {

template <>
struct Runtime<Device::Type::kCpu> : RuntimeBase<Runtime<Device::Type::kCpu>> {
  static constexpr Device::Type kDeviceType = Device::Type::kCpu;
};

static_assert(Runtime<Device::Type::kCpu>::Validate());

}  // namespace infini::ops

#endif
