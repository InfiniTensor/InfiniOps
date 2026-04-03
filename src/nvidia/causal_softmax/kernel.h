#ifndef INFINI_OPS_NVIDIA_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_NVIDIA_CAUSAL_SOFTMAX_KERNEL_H_

#include <utility>

#include "cuda/causal_softmax/kernel.h"
#include "nvidia/caster_.h"
#include "nvidia/device_property.h"

namespace infini::ops {

namespace causal_softmax {

struct NvidiaBackend {
  using stream_t = cudaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kNvidia;

  static int GetOptimalBlockSize() {
    return ComputeOptimalBlockSize(QueryMaxThreadsPerBlock());
  }
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
