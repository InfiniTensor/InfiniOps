#ifndef INFINI_OPS_HYGON_RECURRENT_GATED_DELTA_RULE_KERNEL_H_
#define INFINI_OPS_HYGON_RECURRENT_GATED_DELTA_RULE_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/recurrent_gated_delta_rule/kernel.h"

namespace infini::ops {

template <>
class Operator<RecurrentGatedDeltaRule, Device::Type::kHygon, 0>
    : public CudaRecurrentGatedDeltaRule<Runtime<Device::Type::kHygon>> {
 public:
  using CudaRecurrentGatedDeltaRule<Runtime<Device::Type::kHygon>>::CudaRecurrentGatedDeltaRule;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_RECURRENT_GATED_DELTA_RULE_KERNEL_H_
