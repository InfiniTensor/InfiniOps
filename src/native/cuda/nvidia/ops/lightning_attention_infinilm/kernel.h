#ifndef INFINI_OPS_NVIDIA_LIGHTNING_ATTENTION_INFINILM_KERNEL_H_
#define INFINI_OPS_NVIDIA_LIGHTNING_ATTENTION_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/lightning_attention_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<LightningAttentionInfinilm, Device::Type::kNvidia>
    : public CudaLightningAttentionInfinilm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaLightningAttentionInfinilm<
      Runtime<Device::Type::kNvidia>>::CudaLightningAttentionInfinilm;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_LIGHTNING_ATTENTION_INFINILM_KERNEL_H_
