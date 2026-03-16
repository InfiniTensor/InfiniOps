#ifndef INFINI_OPS_NVIDIA_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_NVIDIA_CAUSAL_SOFTMAX_KERNEL_H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "cuda/causal_softmax/kernel.h"

namespace infini::ops {

namespace causal_softmax {

struct NvidiaBackend {
  static constexpr auto device_value = Device::Type::kNvidia;

  using stream_t = cudaStream_t;
};

}  // namespace causal_softmax

template <>
class Operator<CausalSoftmax, Device::Type::kNvidia>
    : public CudaCausalSoftmax<causal_softmax::NvidiaBackend> {
 public:
  using CudaCausalSoftmax<causal_softmax::NvidiaBackend>::CudaCausalSoftmax;
};

}  // namespace infini::ops

#endif
