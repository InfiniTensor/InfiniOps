#pragma once
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/recurrent_gated_delta_rule/kernel.h"
namespace infini::ops {
template <>
class Operator<RecurrentGatedDeltaRule, Device::Type::kNvidia, 0>
    : public CudaRecurrentGatedDeltaRule<Device::Type::kNvidia> {
 public:
  using CudaRecurrentGatedDeltaRule<
      Device::Type::kNvidia>::CudaRecurrentGatedDeltaRule;
};
}  // namespace infini::ops
