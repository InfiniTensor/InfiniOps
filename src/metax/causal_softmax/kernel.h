#ifndef INFINI_OPS_METAX_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_METAX_CAUSAL_SOFTMAX_KERNEL_H_

#include <utility>

#include "cuda/causal_softmax/kernel.h"
#include "metax/device_.h"

namespace infini::ops {

namespace causal_softmax {

struct MetaxBackend {
  using stream_t = mcStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kMetax;

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
class Operator<CausalSoftmax, Device::Type::kMetax>
    : public CudaCausalSoftmax<causal_softmax::MetaxBackend> {
 public:
  using CudaCausalSoftmax<causal_softmax::MetaxBackend>::CudaCausalSoftmax;
};

}  // namespace infini::ops

#endif
