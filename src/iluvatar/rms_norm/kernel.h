#ifndef INFINI_OPS_ILUVATAR_RMS_NORM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_RMS_NORM_KERNEL_H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "cuda/rms_norm/kernel.h"

namespace infini::ops {

namespace rms_norm {

struct IluvatarBackend {
  using stream_t = cudaStream_t;

  static constexpr auto setDevice = [](int dev) { cudaSetDevice(dev); };

  static constexpr auto streamSynchronize = [](stream_t s) {
    cudaStreamSynchronize(s);
  };
};

}  // namespace rms_norm

template <>
class Operator<RmsNorm, Device::Type::kIluvatar>
    : public CudaRmsNorm<rms_norm::IluvatarBackend> {
 public:
  using CudaRmsNorm<rms_norm::IluvatarBackend>::CudaRmsNorm;
};

}  // namespace infini::ops

#endif
