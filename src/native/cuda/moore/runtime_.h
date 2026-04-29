#ifndef INFINI_OPS_MOORE_RUNTIME_H_
#define INFINI_OPS_MOORE_RUNTIME_H_

#include <musa_runtime.h>

#include <utility>

#include "native/cuda/moore/device_.h"
#include "native/cuda/moore/runtime_utils.h"
#include "native/cuda/runtime_.h"

namespace infini::ops {

template <>
struct Runtime<Device::Type::kMoore>
    : CudaRuntime<Runtime<Device::Type::kMoore>> {
  using Stream = musaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kMoore;

  static constexpr auto Malloc = [](auto&&... args) {
    return musaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = [](auto&&... args) {
    return musaMemcpy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Free = [](auto&&... args) {
    return musaFree(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemcpyHostToDevice = musaMemcpyHostToDevice;

  static constexpr auto MemcpyDeviceToHost = musaMemcpyDeviceToHost;

  static constexpr auto Memset = musaMemset;
};

static_assert(Runtime<Device::Type::kMoore>::Validate());

}  // namespace infini::ops

#endif
