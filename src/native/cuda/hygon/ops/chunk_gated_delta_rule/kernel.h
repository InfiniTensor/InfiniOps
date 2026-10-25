#pragma once
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/chunk_gated_delta_rule/kernel.h"
namespace infini::ops {
template <>
class Operator<ChunkGatedDeltaRule, Device::Type::kHygon, 0>
    : public CudaChunkGatedDeltaRule<Device::Type::kHygon> {
 public:
  using CudaChunkGatedDeltaRule<Device::Type::kHygon>::CudaChunkGatedDeltaRule;
};
}  // namespace infini::ops
