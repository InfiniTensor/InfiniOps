#ifndef INFINI_OPS_METAX_RMS_NORM_KERNEL_H_
#define INFINI_OPS_METAX_RMS_NORM_KERNEL_H_

#include <utility>

#include "cuda/rms_norm/kernel.h"
#include "metax/device_.h"

namespace infini::ops {

namespace rms_norm {

struct MetaxBackend {
  using stream_t = mcStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kMetax;

  static int GetOptimalBlockSize() {
    return ComputeOptimalBlockSize(QueryMaxThreadsPerBlock());
  }
};

}  // namespace rms_norm

template <>
class Operator<RmsNorm, Device::Type::kMetax>
    : public CudaRmsNorm<rms_norm::MetaxBackend> {
 public:
  using CudaRmsNorm<rms_norm::MetaxBackend>::CudaRmsNorm;
};

}  // namespace infini::ops

#endif
