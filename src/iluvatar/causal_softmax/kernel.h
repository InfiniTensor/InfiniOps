#ifndef INFINI_OPS_ILUVATAR_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_ILUVATAR_CAUSAL_SOFTMAX_KERNEL_H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

// clang-format off
#include "iluvatar/device_.h"
// clang-format on

#include "cuda/causal_softmax/kernel.h"

namespace infini::ops {

namespace causal_softmax {

struct IluvatarBackend {
  using stream_t = cudaStream_t;
};

}  // namespace causal_softmax

template <>
class Operator<CausalSoftmax, Device::Type::kIluvatar>
    : public CudaCausalSoftmax<causal_softmax::IluvatarBackend> {
 public:
  using CudaCausalSoftmax<causal_softmax::IluvatarBackend>::CudaCausalSoftmax;
};

}  // namespace infini::ops

#endif
