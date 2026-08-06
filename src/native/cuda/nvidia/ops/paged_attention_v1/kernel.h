#ifndef INFINI_OPS_NVIDIA_PAGED_ATTENTION_V1_KERNEL_H_
#define INFINI_OPS_NVIDIA_PAGED_ATTENTION_V1_KERNEL_H_

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/paged_attention_v1/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedAttentionV1, Device::Type::kNvidia>
    : public CudaPagedAttentionV1<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaPagedAttentionV1<
      Runtime<Device::Type::kNvidia>>::CudaPagedAttentionV1;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_PAGED_ATTENTION_V1_KERNEL_H_
