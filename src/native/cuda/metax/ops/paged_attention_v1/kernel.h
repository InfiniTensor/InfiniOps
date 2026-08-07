#ifndef INFINI_OPS_METAX_PAGED_ATTENTION_V1_KERNEL_H_
#define INFINI_OPS_METAX_PAGED_ATTENTION_V1_KERNEL_H_

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/paged_attention_v1/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedAttentionV1, Device::Type::kMetax>
    : public CudaPagedAttentionV1<Runtime<Device::Type::kMetax>> {
 public:
  using CudaPagedAttentionV1<
      Runtime<Device::Type::kMetax>>::CudaPagedAttentionV1;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_METAX_PAGED_ATTENTION_V1_KERNEL_H_
