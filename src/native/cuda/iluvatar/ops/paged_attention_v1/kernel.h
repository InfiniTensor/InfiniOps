#ifndef INFINI_OPS_ILUVATAR_PAGED_ATTENTION_V1_KERNEL_H_
#define INFINI_OPS_ILUVATAR_PAGED_ATTENTION_V1_KERNEL_H_

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/paged_attention_v1/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedAttentionV1, Device::Type::kIluvatar>
    : public CudaPagedAttentionV1<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaPagedAttentionV1<
      Runtime<Device::Type::kIluvatar>>::CudaPagedAttentionV1;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ILUVATAR_PAGED_ATTENTION_V1_KERNEL_H_
