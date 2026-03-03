#ifndef INFINI_OPS_ILUVATAR_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_ILUVATAR_CAUSAL_SOFTMAX_KERNEL_H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "cuda/causal_softmax/kernel.h"

namespace infini::ops {

namespace causal_softmax {

struct IluvatarBackend {
  using stream_t = cudaStream_t;

  static constexpr auto setDevice = [](int dev) { cudaSetDevice(dev); };

  static constexpr auto streamSynchronize = [](stream_t s) {
    cudaStreamSynchronize(s);
  };
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
