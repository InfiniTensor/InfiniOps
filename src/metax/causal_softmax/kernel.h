#ifndef INFINI_OPS_METAX_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_METAX_CAUSAL_SOFTMAX_KERNEL_H_

#include <utility>

// clang-format off
#include <mcr/mc_runtime.h>
// clang-format on

#include "cuda/causal_softmax/kernel.h"

namespace infini::ops {

namespace causal_softmax {

struct MetaxBackend {
  using stream_t = mcStream_t;
};

}  // namespace causal_softmax

template <>
class Operator<CausalSoftmax, Device::Type::kMetax>
    : public CudaCausalSoftmax<causal_softmax::MetaxBackend> {
 public:
  using CudaCausalSoftmax<causal_softmax::MetaxBackend>::CudaCausalSoftmax;
};

}  // namespace infini::ops

#endif
