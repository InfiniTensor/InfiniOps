#ifndef INFINI_OPS_HYGON_RUNTIME_H_
#define INFINI_OPS_HYGON_RUNTIME_H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "native/cuda/hygon/device_.h"
#include "native/cuda/hygon/runtime_utils.h"
#include "native/cuda/runtime_.h"

namespace infini::ops {

template <>
struct Runtime<Device::Type::kHygon>
    : CudaRuntime<Runtime<Device::Type::kHygon>> {
  using Stream = cudaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kHygon;

  static constexpr auto Malloc = [](auto&&... args) {
    return cudaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = cudaMemcpy;

  static constexpr auto Free = [](auto&&... args) {
    return cudaFree(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemcpyHostToDevice = cudaMemcpyHostToDevice;

  static constexpr auto MemcpyDeviceToHost = cudaMemcpyDeviceToHost;

  static constexpr auto Memset = cudaMemset;
};

static_assert(Runtime<Device::Type::kHygon>::Validate());

}  // namespace infini::ops

#endif
