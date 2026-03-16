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
  static constexpr auto device_value = Device::Type::kIluvatar;

  using stream_t = cudaStream_t;
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
