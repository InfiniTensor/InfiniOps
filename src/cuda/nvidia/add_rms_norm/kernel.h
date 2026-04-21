#ifndef INFINI_OPS_NVIDIA_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_NVIDIA_ADD_RMS_NORM_KERNEL_H_

#include <utility>

#include "cuda/add_rms_norm/kernel.h"
#include "cuda/nvidia/caster.cuh"
#include "cuda/nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<AddRmsNorm, Device::Type::kNvidia>
    : public CudaAddRmsNorm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaAddRmsNorm<Runtime<Device::Type::kNvidia>>::CudaAddRmsNorm;
};

}  // namespace infini::ops

#endif
