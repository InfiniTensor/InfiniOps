#ifndef INFINI_OPS_QY_RMS_NORM_KERNEL_H_
#define INFINI_OPS_QY_RMS_NORM_KERNEL_H_

#include "native/cuda/ops/rms_norm/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kQy>
    : public CudaRmsNorm<Runtime<Device::Type::kQy>> {
 public:
  using CudaRmsNorm<Runtime<Device::Type::kQy>>::CudaRmsNorm;
};

}  // namespace infini::ops

#endif
