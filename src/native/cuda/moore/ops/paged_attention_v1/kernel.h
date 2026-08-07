#ifndef INFINI_OPS_MOORE_PAGED_ATTENTION_V1_KERNEL_H_
#define INFINI_OPS_MOORE_PAGED_ATTENTION_V1_KERNEL_H_

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/paged_attention_v1/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedAttentionV1, Device::Type::kMoore>
    : public CudaPagedAttentionV1<Runtime<Device::Type::kMoore>> {
 public:
  using CudaPagedAttentionV1<
      Runtime<Device::Type::kMoore>>::CudaPagedAttentionV1;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_MOORE_PAGED_ATTENTION_V1_KERNEL_H_
