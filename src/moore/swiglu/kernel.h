#ifndef INFINI_OPS_MOORE_SWIGLU_KERNEL_H_
#define INFINI_OPS_MOORE_SWIGLU_KERNEL_H_

#include <utility>

// clang-format off
#include "moore/polyfills.cuh"
// clang-format on

#include "cuda/swiglu/kernel.h"
#include "moore/device_.h"

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
    int max_threads = QueryMaxThreadsPerBlock();
    if (max_threads >= 2048) return 2048;
    if (max_threads >= 1024) return 1024;
    if (max_threads >= 512) return 512;
    if (max_threads >= 256) return 256;
    return 128;
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
