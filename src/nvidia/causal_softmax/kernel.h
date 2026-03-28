#ifndef INFINI_OPS_NVIDIA_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_NVIDIA_CAUSAL_SOFTMAX_KERNEL_H_

#include <utility>

#include "cuda/causal_softmax/kernel.h"
#include "nvidia/device_.h"

namespace infini::ops {

namespace causal_softmax {

struct NvidiaBackend {
  using stream_t = cudaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kNvidia;

  static int GetOptimalBlockSize() {
    int max_threads = QueryMaxThreadsPerBlock();
    if (max_threads >= 2048) return 2048;
    if (max_threads >= 1024) return 1024;
    if (max_threads >= 512) return 512;
    if (max_threads >= 256) return 256;
    return 128;
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
