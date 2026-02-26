#ifndef INFINI_OPS_NVIDIA_ADD_KERNEL_H_
#define INFINI_OPS_NVIDIA_ADD_KERNEL_H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "cuda/add/kernel.h"

namespace infini::ops {

struct NvidiaBackend {
  using stream_t = cudaStream_t;

  static constexpr auto malloc = cudaMalloc;
  static constexpr auto memcpy = cudaMemcpy;
  static constexpr auto free = cudaFree;
  static constexpr auto MemcpyH2D = cudaMemcpyHostToDevice;
};

template <>
class Operator<Add, Device::Type::kNvidia> : public CudaAdd<NvidiaBackend> {
 public:
  using CudaAdd<NvidiaBackend>::CudaAdd;
};

}  // namespace infini::ops

#endif
