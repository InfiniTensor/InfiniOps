#ifndef INFINI_OPS_NVIDIA_RMS_NORM_KERNEL_H_
#define INFINI_OPS_NVIDIA_RMS_NORM_KERNEL_H_

#include <utility>

#include "cuda/rms_norm/kernel.h"
#include "nvidia/device_property.h"

namespace infini::ops {

namespace rms_norm {

struct NvidiaBackend {
  using stream_t = cudaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kNvidia;

  static int GetOptimalBlockSize() {
    return ComputeOptimalBlockSize(QueryMaxThreadsPerBlock());
  }
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
