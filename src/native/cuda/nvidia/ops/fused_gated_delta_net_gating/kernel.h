#ifndef INFINI_OPS_NVIDIA_FUSED_GATED_DELTA_NET_GATING_KERNEL_H_
#define INFINI_OPS_NVIDIA_FUSED_GATED_DELTA_NET_GATING_KERNEL_H_

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/fused_gated_delta_net_gating/kernel.h"

namespace infini::ops {

template <>
class Operator<FusedGatedDeltaNetGating, Device::Type::kNvidia>
    : public CudaFusedGatedDeltaNetGating<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaFusedGatedDeltaNetGating<Runtime<Device::Type::kNvidia>>::CudaFusedGatedDeltaNetGating;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_FUSED_GATED_DELTA_NET_GATING_KERNEL_H_
