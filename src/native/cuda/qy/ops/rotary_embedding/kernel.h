#ifndef INFINI_OPS_QY_ROTARY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_QY_ROTARY_EMBEDDING_KERNEL_H_

#include "native/cuda/ops/rotary_embedding/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
class Operator<RotaryEmbedding, Device::Type::kQy>
    : public CudaRotaryEmbedding<Runtime<Device::Type::kQy>> {
 public:
  using CudaRotaryEmbedding<Runtime<Device::Type::kQy>>::CudaRotaryEmbedding;
};

}  // namespace infini::ops

#endif
