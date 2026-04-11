#ifndef INFINI_OPS_NVIDIA_ROTARY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_NVIDIA_ROTARY_EMBEDDING_KERNEL_H_

#include <utility>

#include "cuda/rotary_embedding/kernel.h"
#include "nvidia/caster.cuh"
#include "nvidia/rotary_embedding/registry.h"
#include "nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<RotaryEmbedding, Device::Type::kNvidia>
    : public CudaRotaryEmbedding<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaRotaryEmbedding<Runtime<Device::Type::kNvidia>>::CudaRotaryEmbedding;
};

}  // namespace infini::ops

#endif
