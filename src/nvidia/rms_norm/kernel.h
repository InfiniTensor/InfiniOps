#ifndef INFINI_OPS_NVIDIA_RMS_NORM_KERNEL_H_
#define INFINI_OPS_NVIDIA_RMS_NORM_KERNEL_H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "cuda/rms_norm/kernel.h"

namespace infini::ops {

namespace rms_norm {

struct NvidiaBackend {
  using stream_t = cudaStream_t;
};

}  // namespace rms_norm

template <>
class Operator<RmsNorm, Device::Type::kNvidia>
    : public CudaRmsNorm<rms_norm::NvidiaBackend> {
 public:
  using CudaRmsNorm<rms_norm::NvidiaBackend>::CudaRmsNorm;
};

}  // namespace infini::ops

#endif
