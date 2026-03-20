#ifndef INFINI_OPS_HYGON_ADD_KERNEL_H_
#define INFINI_OPS_HYGON_ADD_KERNEL_H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "cuda/add/kernel.h"

namespace infini::ops {

namespace add {

struct HygonBackend {
  using stream_t = cudaStream_t;

  static constexpr auto malloc = [](auto&&... args) {
    return cudaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto memcpy = cudaMemcpy;

  static constexpr auto free = [](auto&&... args) {
    return cudaFree(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto memcpyH2D = cudaMemcpyHostToDevice;
};

}  // namespace add

template <>
class Operator<Add, Device::Type::kHygon> : public CudaAdd<add::HygonBackend> {
 public:
  using CudaAdd<add::HygonBackend>::CudaAdd;
};

}  // namespace infini::ops

#endif
