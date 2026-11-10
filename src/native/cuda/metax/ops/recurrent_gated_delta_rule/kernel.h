#ifndef INFINI_OPS_METAX_RECURRENT_GATED_DELTA_RULE_KERNEL_H_
#define INFINI_OPS_METAX_RECURRENT_GATED_DELTA_RULE_KERNEL_H_

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/recurrent_gated_delta_rule/kernel.h"

namespace infini::ops {

// MetaX provider for the shared CUDA single-token decode gated-delta-rule
// kernel. The shared host path performs all shape/extent validation and
// drives the same device code the NVIDIA provider runs; MetaX adds no
// extra checks (no CUDA device query is available here).
template <>
class Operator<RecurrentGatedDeltaRule, Device::Type::kMetax>
    : public CudaRecurrentGatedDeltaRule<Runtime<Device::Type::kMetax>> {
 public:
  using CudaRecurrentGatedDeltaRule<
      Runtime<Device::Type::kMetax>>::CudaRecurrentGatedDeltaRule;
};

}  // namespace infini::ops

#endif
