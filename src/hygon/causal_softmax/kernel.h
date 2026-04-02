#ifndef INFINI_OPS_HYGON_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_HYGON_CAUSAL_SOFTMAX_KERNEL_H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

// clang-format off
#include "hygon/device_.h"
// clang-format on

#include "cuda/causal_softmax/kernel.h"

namespace infini::ops {

namespace causal_softmax {

struct HygonBackend {
  using stream_t = cudaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kHygon;

  static constexpr int max_block_size = 256;

  static int GetOptimalBlockSize() {
    return ComputeOptimalBlockSize(QueryMaxThreadsPerBlock());
  }
};

}  // namespace causal_softmax

template <>
class Operator<CausalSoftmax, Device::Type::kHygon>
    : public CudaCausalSoftmax<causal_softmax::HygonBackend> {
 public:
  using CudaCausalSoftmax<causal_softmax::HygonBackend>::CudaCausalSoftmax;
};

}  // namespace infini::ops

#endif
