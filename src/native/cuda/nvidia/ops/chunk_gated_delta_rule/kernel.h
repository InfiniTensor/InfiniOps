#pragma once
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/chunk_gated_delta_rule/kernel.h"
namespace infini::ops {
template <>
class Operator<ChunkGatedDeltaRule, Device::Type::kNvidia, 0>
    : public CudaChunkGatedDeltaRule<Device::Type::kNvidia> {
 public:
  using CudaChunkGatedDeltaRule<Device::Type::kNvidia>::CudaChunkGatedDeltaRule;
};
}  // namespace infini::ops
