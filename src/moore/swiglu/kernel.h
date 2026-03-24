#ifndef INFINI_OPS_MOORE_SWIGLU_KERNEL_H_
#define INFINI_OPS_MOORE_SWIGLU_KERNEL_H_

#include <utility>

#include "cuda/swiglu/kernel.h"

namespace infini::ops {

namespace swiglu {

struct MooreBackend {
  using stream_t = musaStream_t;

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
