#ifndef INFINI_OPS_QY_FUSED_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_QY_FUSED_ADD_RMS_NORM_KERNEL_H_

#include "native/cuda/ops/fused_add_rms_norm/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
class Operator<FusedAddRmsNorm, Device::Type::kQy>
    : public CudaFusedAddRmsNorm<Runtime<Device::Type::kQy>> {
 public:
  using CudaFusedAddRmsNorm<Runtime<Device::Type::kQy>>::CudaFusedAddRmsNorm;
};

}  // namespace infini::ops

#endif
