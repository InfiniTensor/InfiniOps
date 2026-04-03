#ifndef INFINI_OPS_ILUVATAR_RUNTIME_H_
#define INFINI_OPS_ILUVATAR_RUNTIME_H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "cuda/runtime.h"
#include "iluvatar/device_.h"
#include "iluvatar/runtime_utils.h"

namespace infini::ops {

template <>
struct Runtime<Device::Type::kIluvatar>
    : CudaRuntime<Runtime<Device::Type::kIluvatar>> {
  using Stream = cudaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kIluvatar;

  static constexpr auto Malloc = [](auto&&... args) {
    return cudaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = cudaMemcpy;

  static constexpr auto Free = cudaFree;

  static constexpr auto MemcpyHostToDevice = cudaMemcpyHostToDevice;

  static constexpr auto MemcpyDeviceToHost = cudaMemcpyDeviceToHost;

  static constexpr auto Memset = cudaMemset;
};

static_assert(Runtime<Device::Type::kIluvatar>::Validate());

}  // namespace infini::ops

#endif
