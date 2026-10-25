#pragma once
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/recurrent_gated_delta_rule/kernel.h"
namespace infini::ops {
template <>
class Operator<RecurrentGatedDeltaRule, Device::Type::kHygon, 0>
    : public CudaRecurrentGatedDeltaRule<Device::Type::kHygon> {
 public:
  using CudaRecurrentGatedDeltaRule<
      Device::Type::kHygon>::CudaRecurrentGatedDeltaRule;
};
}  // namespace infini::ops
