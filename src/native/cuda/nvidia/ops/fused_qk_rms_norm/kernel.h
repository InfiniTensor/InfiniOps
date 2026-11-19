#ifndef INFINI_OPS_NVIDIA_FUSED_QK_RMS_NORM_KERNEL_H_
#define INFINI_OPS_NVIDIA_FUSED_QK_RMS_NORM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/fused_qk_rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<FusedQkRmsNorm, Device::Type::kNvidia>
    : public CudaFusedQkRmsNorm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaFusedQkRmsNorm<Runtime<Device::Type::kNvidia>>::CudaFusedQkRmsNorm;
  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif
