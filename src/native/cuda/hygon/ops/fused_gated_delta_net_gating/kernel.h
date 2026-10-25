#pragma once
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/fused_gated_delta_net_gating/kernel.h"
namespace infini::ops {
template <>
class Operator<FusedGatedDeltaNetGating, Device::Type::kHygon, 0>
    : public CudaFusedGatedDeltaNetGating<Device::Type::kHygon> {
 public:
  using CudaFusedGatedDeltaNetGating<
      Device::Type::kHygon>::CudaFusedGatedDeltaNetGating;
};
}  // namespace infini::ops
