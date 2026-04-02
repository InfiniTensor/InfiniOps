#ifndef INFINI_OPS_NVIDIA_RUNTIME_H_
#define INFINI_OPS_NVIDIA_RUNTIME_H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "nvidia/device_.h"
#include "runtime.h"

namespace infini::ops {

template <>
struct Runtime<Device::Type::kNvidia>
    : CudaLikeRuntime<Runtime<Device::Type::kNvidia>> {
  using Stream = cudaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kNvidia;

  static constexpr auto Malloc = [](auto&&... args) {
    return cudaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = cudaMemcpy;

  static constexpr auto Free = cudaFree;

  static constexpr auto MemcpyHostToDevice = cudaMemcpyHostToDevice;

  static int GetOptimalBlockSize() {
    int max_threads = QueryMaxThreadsPerBlock();
    if (max_threads >= 2048) return 2048;
    if (max_threads >= 1024) return 1024;
    if (max_threads >= 512) return 512;
    if (max_threads >= 256) return 256;
    return 128;
  }
};

static_assert(Runtime<Device::Type::kNvidia>::Validate());

}  // namespace infini::ops

#endif
