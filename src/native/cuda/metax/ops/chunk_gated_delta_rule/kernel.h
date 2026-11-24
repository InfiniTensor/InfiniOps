#ifndef INFINI_OPS_METAX_CHUNK_GATED_DELTA_RULE_KERNEL_H_
#define INFINI_OPS_METAX_CHUNK_GATED_DELTA_RULE_KERNEL_H_

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/chunk_gated_delta_rule/kernel.h"

namespace infini::ops {

// MetaX provider for the shared CUDA chunked-prefill gated-delta-rule
// kernel. The shared host path performs all shape/extent/workspace
// validation and drives the same device code the NVIDIA provider runs;
// MetaX adds no extra checks (no CUDA device query is available here).
template <>
class Operator<ChunkGatedDeltaRule, Device::Type::kMetax>
    : public CudaChunkGatedDeltaRule<Runtime<Device::Type::kMetax>> {
 public:
  using CudaChunkGatedDeltaRule<
      Runtime<Device::Type::kMetax>>::CudaChunkGatedDeltaRule;
};

}  // namespace infini::ops

#endif
