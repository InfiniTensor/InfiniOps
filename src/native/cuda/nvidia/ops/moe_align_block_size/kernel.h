#ifndef INFINI_OPS_NVIDIA_MOE_ALIGN_BLOCK_SIZE_KERNEL_H_
#define INFINI_OPS_NVIDIA_MOE_ALIGN_BLOCK_SIZE_KERNEL_H_

#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/moe_align_block_size/kernel.h"

namespace infini::ops {

template <>
class Operator<MoeAlignBlockSize, Device::Type::kNvidia, 0>
    : public CudaMoeAlignBlockSize<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaMoeAlignBlockSize<Runtime<Device::Type::kNvidia>>::CudaMoeAlignBlockSize;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_MOE_ALIGN_BLOCK_SIZE_KERNEL_H_
