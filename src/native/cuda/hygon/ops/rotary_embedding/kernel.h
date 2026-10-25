#ifndef INFINI_OPS_HYGON_ROTARY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_HYGON_ROTARY_EMBEDDING_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/rotary_embedding/kernel.h"

namespace infini::ops {

template <>
class Operator<RotaryEmbedding, Device::Type::kHygon>
    : public CudaRotaryEmbedding<Runtime<Device::Type::kHygon>> {
 public:
  using CudaRotaryEmbedding<Runtime<Device::Type::kHygon>>::CudaRotaryEmbedding;
  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif
