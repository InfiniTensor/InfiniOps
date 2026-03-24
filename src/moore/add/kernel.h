#ifndef INFINI_OPS_MOORE_ADD_KERNEL_H_
#define INFINI_OPS_MOORE_ADD_KERNEL_H_

#include <utility>

#include "cuda/add/kernel.h"

namespace infini::ops {

namespace add {

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

}  // namespace add

template <>
class Operator<Add, Device::Type::kMoore> : public CudaAdd<add::MooreBackend> {
 public:
  using CudaAdd<add::MooreBackend>::CudaAdd;
};

}  // namespace infini::ops

#endif
