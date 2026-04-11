#ifndef INFINI_OPS_NVIDIA_RMS_NORM_KERNEL_H_
#define INFINI_OPS_NVIDIA_RMS_NORM_KERNEL_H_

#include <utility>

#include "cuda/rms_norm/kernel.h"
#include "nvidia/caster.cuh"
#include "nvidia/rms_norm/registry.h"
#include "nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kNvidia>
    : public CudaRmsNorm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaRmsNorm<Runtime<Device::Type::kNvidia>>::CudaRmsNorm;
};

}  // namespace infini::ops

#endif
