#ifndef INFINI_OPS_MOORE_PAGED_ATTENTION_PREFILL_INFINILM_KERNEL_H_
#define INFINI_OPS_MOORE_PAGED_ATTENTION_PREFILL_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include <infini/rt/moore/runtime_.h>
#include "native/cuda/moore/runtime_utils.h"
#include "native/cuda/ops/paged_attention_prefill_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedAttentionPrefillInfinilm, Device::Type::kMoore>
    : public CudaPagedAttentionPrefillInfinilm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaPagedAttentionPrefillInfinilm<
      Runtime<Device::Type::kMoore>>::CudaPagedAttentionPrefillInfinilm;
};

}  // namespace infini::ops

#endif
