#ifndef INFINI_OPS_ILUVATAR_RMS_NORM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_RMS_NORM_KERNEL_H_

#include <utility>

#include "cuda/rms_norm/kernel.h"
#include "iluvatar/device_.h"

namespace infini::ops {

namespace rms_norm {

struct IluvatarBackend {
  using stream_t = cudaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kIluvatar;

  static int GetOptimalBlockSize() {
    int max_threads = QueryMaxThreadsPerBlock();
    if (max_threads >= 2048) return 2048;
    if (max_threads >= 1024) return 1024;
    if (max_threads >= 512) return 512;
    if (max_threads >= 256) return 256;
    return 128;
  }
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
