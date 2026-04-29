#ifndef INFINI_OPS_NVIDIA_RMS_NORM_KERNEL_H_
#define INFINI_OPS_NVIDIA_RMS_NORM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kNvidia>
    : public CudaRmsNorm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaRmsNorm<Runtime<Device::Type::kNvidia>>::CudaRmsNorm;
};

}  // namespace infini::ops

#endif
