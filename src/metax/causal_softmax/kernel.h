#ifndef INFINI_OPS_METAX_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_METAX_CAUSAL_SOFTMAX_KERNEL_H_

#include <utility>

#include "cuda/causal_softmax/kernel.h"
#include "metax/caster_.h"
#include "metax/device_property.h"

namespace infini::ops {

namespace causal_softmax {

struct MetaxBackend {
  using stream_t = mcStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kMetax;

  static int GetOptimalBlockSize() {
    return ComputeOptimalBlockSize(QueryMaxThreadsPerBlock());
  }
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
