#ifndef INFINI_OPS_CAMBRICON_RUNTIME_H_
#define INFINI_OPS_CAMBRICON_RUNTIME_H_

#include <cnrt.h>

#include "native/cambricon/device_.h"
#include "runtime.h"

namespace infini::ops {

template <>
struct Runtime<Device::Type::kCambricon>
    : DeviceRuntime<Runtime<Device::Type::kCambricon>> {
  using Stream = cnrtQueue_t;

  static constexpr Device::Type kDeviceType = Device::Type::kCambricon;

  static constexpr auto Malloc = cnrtMalloc;

  static constexpr auto Free = cnrtFree;

  static constexpr auto Memcpy = cnrtMemcpy;

  static constexpr auto MemcpyHostToDevice = cnrtMemcpyHostToDev;

  static constexpr auto MemcpyDeviceToHost = cnrtMemcpyDevToHost;

  static constexpr auto Memset = cnrtMemset;
};

static_assert(Runtime<Device::Type::kCambricon>::Validate());

}  // namespace infini::ops

#endif
