#ifndef INFINI_OPS_HYGON_RMS_NORM_KERNEL_H_
#define INFINI_OPS_HYGON_RMS_NORM_KERNEL_H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

// clang-format off
#include "hygon/device_.h"
// clang-format on

#include "cuda/rms_norm/kernel.h"

namespace infini::ops {

namespace rms_norm {

struct HygonBackend {
  using stream_t = cudaStream_t;

  static constexpr int max_block_size = CUDA_BLOCK_SIZE_256;
};

}  // namespace rms_norm

template <>
class Operator<RmsNorm, Device::Type::kHygon>
    : public CudaRmsNorm<rms_norm::HygonBackend> {
 public:
  using CudaRmsNorm<rms_norm::HygonBackend>::CudaRmsNorm;
};

}  // namespace infini::ops

#endif
