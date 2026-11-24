#ifndef INFINI_OPS_HYGON_FUSED_GATED_DELTA_NET_GATING_KERNEL_H_
#define INFINI_OPS_HYGON_FUSED_GATED_DELTA_NET_GATING_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/fused_gated_delta_net_gating/kernel.h"

namespace infini::ops {

template <>
class Operator<FusedGatedDeltaNetGating, Device::Type::kHygon>
    : public CudaFusedGatedDeltaNetGating<Runtime<Device::Type::kHygon>> {
 public:
  using CudaFusedGatedDeltaNetGating<Runtime<Device::Type::kHygon>>::CudaFusedGatedDeltaNetGating;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_FUSED_GATED_DELTA_NET_GATING_KERNEL_H_
