#ifndef INFINI_OPS_METAX_RUNTIME_H_
#define INFINI_OPS_METAX_RUNTIME_H_

#include <mcr/mc_runtime.h>

#include "metax/device_.h"
#include "runtime.h"

namespace infini::ops {

template <>
struct Runtime<Device::Type::kMetax>
    : CudaLikeRuntime<Runtime<Device::Type::kMetax>> {
  using Stream = mcStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kMetax;

  static constexpr auto Malloc = mcMalloc;

  static constexpr auto Memcpy = mcMemcpy;

  static constexpr auto Free = mcFree;

  static constexpr auto MemcpyHostToDevice = mcMemcpyHostToDevice;

  static int GetOptimalBlockSize() {
    int max_threads = QueryMaxThreadsPerBlock();
    if (max_threads >= 2048) return 2048;
    if (max_threads >= 1024) return 1024;
    if (max_threads >= 512) return 512;
    if (max_threads >= 256) return 256;
    return 128;
  }
};

static_assert(Runtime<Device::Type::kMetax>::Validate());

}  // namespace infini::ops

#endif
