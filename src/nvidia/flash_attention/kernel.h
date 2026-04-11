#ifndef INFINI_OPS_NVIDIA_FLASH_ATTENTION_KERNEL_H_
#define INFINI_OPS_NVIDIA_FLASH_ATTENTION_KERNEL_H_

#include "cuda/flash_attention/kernel.h"
#include "nvidia/caster.cuh"
#include "nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<FlashAttention, Device::Type::kNvidia>
    : public CudaFlashAttention<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaFlashAttention<Runtime<Device::Type::kNvidia>>::CudaFlashAttention;
};

}  // namespace infini::ops

#endif
