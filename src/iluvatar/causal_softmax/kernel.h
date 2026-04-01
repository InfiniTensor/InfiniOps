#ifndef INFINI_OPS_ILUVATAR_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_ILUVATAR_CAUSAL_SOFTMAX_KERNEL_H_

#include <utility>

#include "cuda/causal_softmax/kernel.h"
#include "iluvatar/device_.h"

namespace infini::ops {

namespace causal_softmax {

struct IluvatarBackend {
  using stream_t = cudaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kIluvatar;

  static int GetOptimalBlockSize() {
    return ComputeOptimalBlockSize(QueryMaxThreadsPerBlock());
  }
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
