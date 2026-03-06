#ifndef INFINI_OPS_NVIDIA_RUNTIME_H_
#define INFINI_OPS_NVIDIA_RUNTIME_H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "runtime.h"

namespace infini::ops {

template <>
struct Runtime<Device::Type::kNvidia> {
  using Stream = cudaStream_t;

  static constexpr auto Malloc = [](auto&&... args) {
    return cudaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = cudaMemcpy;

  static constexpr auto Free = cudaFree;

  static constexpr auto MemcpyHostToDevice = cudaMemcpyHostToDevice;
};

}  // namespace infini::ops

#endif
