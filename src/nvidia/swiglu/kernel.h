#ifndef INFINI_OPS_NVIDIA_SWIGLU_KERNEL_H_
#define INFINI_OPS_NVIDIA_SWIGLU_KERNEL_H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "cuda/swiglu/kernel.h"

namespace infini::ops {

namespace swiglu {

struct NvidiaBackend {
  static constexpr auto device_value = Device::Type::kNvidia;

  using stream_t = cudaStream_t;

  static constexpr auto malloc = [](auto&&... args) {
    return cudaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto memcpy = cudaMemcpy;

  static constexpr auto free = cudaFree;

  static constexpr auto memcpyH2D = cudaMemcpyHostToDevice;
};

}  // namespace swiglu

template <>
class Operator<Swiglu, Device::Type::kNvidia>
    : public CudaSwiglu<swiglu::NvidiaBackend> {
 public:
  using CudaSwiglu<swiglu::NvidiaBackend>::CudaSwiglu;
};

}  // namespace infini::ops

#endif
