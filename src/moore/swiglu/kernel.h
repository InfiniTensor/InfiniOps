#ifndef INFINI_OPS_MOORE_SWIGLU_KERNEL_H_
#define INFINI_OPS_MOORE_SWIGLU_KERNEL_H_

#include <utility>

// clang-format off
#include "moore/polyfills.cuh"
// clang-format on

#include "cuda/swiglu/kernel.h"
#include "moore/data_type_.h"
#include "moore/device_property.h"

namespace infini::ops {

namespace swiglu {

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

}  // namespace swiglu

template <>
class Operator<Swiglu, Device::Type::kMoore>
    : public CudaSwiglu<swiglu::MooreBackend> {
 public:
  using CudaSwiglu<swiglu::MooreBackend>::CudaSwiglu;
};

}  // namespace infini::ops

#endif
