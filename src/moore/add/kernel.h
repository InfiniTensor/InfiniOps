#ifndef INFINI_OPS_MOORE_ADD_KERNEL_H_
#define INFINI_OPS_MOORE_ADD_KERNEL_H_

#include <utility>

// clang-format off
#include "moore/polyfills.cuh"
// clang-format on

#include "cuda/add/kernel.h"
#include "moore/device_.h"

namespace infini::ops {

namespace add {

struct MooreBackend {
  using stream_t = musaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kMoore;

  static constexpr auto malloc = [](auto&&... args) {
    return musaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto memcpy = [](auto&&... args) {
    return musaMemcpy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto free = [](auto&&... args) {
    return musaFree(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto memcpyH2D = musaMemcpyHostToDevice;

  static int GetOptimalBlockSize() {
    return ComputeOptimalBlockSize(QueryMaxThreadsPerBlock());
  }
};

}  // namespace add

template <>
class Operator<Add, Device::Type::kMoore> : public CudaAdd<add::MooreBackend> {
 public:
  using CudaAdd<add::MooreBackend>::CudaAdd;
};

}  // namespace infini::ops

#endif
