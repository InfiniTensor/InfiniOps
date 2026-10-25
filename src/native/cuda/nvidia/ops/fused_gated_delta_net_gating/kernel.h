#pragma once
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/fused_gated_delta_net_gating/kernel.h"
namespace infini::ops {
template <>
class Operator<FusedGatedDeltaNetGating, Device::Type::kNvidia, 0>
    : public CudaFusedGatedDeltaNetGating<Device::Type::kNvidia> {
 public:
  using CudaFusedGatedDeltaNetGating<
      Device::Type::kNvidia>::CudaFusedGatedDeltaNetGating;
};
}  // namespace infini::ops
