#ifndef INFINI_OPS_METAX_RMS_NORM_KERNEL_H_
#define INFINI_OPS_METAX_RMS_NORM_KERNEL_H_

#include <utility>

// clang-format off
#include <mcr/mc_runtime.h>
// clang-format on

#include "cuda/rms_norm/kernel.h"

namespace infini::ops {

namespace rms_norm {

struct MetaxBackend {
  using stream_t = mcStream_t;
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
