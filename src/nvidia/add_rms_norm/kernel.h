#ifndef INFINI_OPS_NVIDIA_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_NVIDIA_ADD_RMS_NORM_KERNEL_H_

#include <utility>

#include "cuda/add_rms_norm/kernel.h"
#include "nvidia/add/kernel.h"
#include "nvidia/add_rms_norm/registry.h"
#include "nvidia/caster.cuh"
#include "nvidia/rms_norm/kernel.h"
#include "nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<AddRmsNorm, Device::Type::kNvidia, 0>
    : public CudaAddRmsNorm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaAddRmsNorm<Runtime<Device::Type::kNvidia>>::CudaAddRmsNorm;
};

template <>
class Operator<AddRmsNorm, Device::Type::kNvidia, 1>
    : public CudaAddRmsNormFused<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaAddRmsNormFused<
      Runtime<Device::Type::kNvidia>>::CudaAddRmsNormFused;
};

}  // namespace infini::ops

#endif
