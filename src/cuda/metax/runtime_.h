#ifndef INFINI_OPS_METAX_RUNTIME_H_
#define INFINI_OPS_METAX_RUNTIME_H_

#include <mcr/mc_runtime.h>

#include "cuda/runtime.h"
#include "metax/device_.h"
#include "metax/runtime_utils.h"

namespace infini::ops {

template <>
struct Runtime<Device::Type::kMetax>
    : CudaRuntime<Runtime<Device::Type::kMetax>> {
  using Stream = mcStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kMetax;

  static constexpr auto Malloc = mcMalloc;

  static constexpr auto Memcpy = mcMemcpy;

  static constexpr auto Free = mcFree;

  static constexpr auto MemcpyHostToDevice = mcMemcpyHostToDevice;

  static constexpr auto MemcpyDeviceToHost = mcMemcpyDeviceToHost;

  static constexpr auto Memset = mcMemset;
};

static_assert(Runtime<Device::Type::kMetax>::Validate());

}  // namespace infini::ops

#endif
