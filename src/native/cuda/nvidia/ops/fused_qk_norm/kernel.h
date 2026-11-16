#ifndef INFINI_OPS_NVIDIA_FUSED_QK_NORM_KERNEL_H_
#define INFINI_OPS_NVIDIA_FUSED_QK_NORM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/fused_qk_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<FusedQkNorm, Device::Type::kNvidia>
    : public CudaFusedQkNorm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaFusedQkNorm<Runtime<Device::Type::kNvidia>>::CudaFusedQkNorm;
  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif
