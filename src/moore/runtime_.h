#ifndef INFINI_OPS_MOORE_RUNTIME_H_
#define INFINI_OPS_MOORE_RUNTIME_H_

#include <musa_runtime.h>

#include <utility>

#include "cuda/runtime.h"
#include "moore/device_.h"
#include "moore/runtime_utils.h"

namespace infini::ops {

template <>
struct Runtime<Device::Type::kMoore>
    : CudaLikeRuntime<Runtime<Device::Type::kMoore>> {
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
};

static_assert(Runtime<Device::Type::kMoore>::Validate());

}  // namespace infini::ops

#endif
