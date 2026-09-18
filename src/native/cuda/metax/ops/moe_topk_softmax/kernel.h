#ifndef INFINI_OPS_METAX_MOE_TOPK_SOFTMAX_KERNEL_H_
#define INFINI_OPS_METAX_MOE_TOPK_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/moe_topk_softmax/kernel.h"

namespace infini::ops {

template <>
class Operator<MoeTopkSoftmax, Device::Type::kMetax>
    : public CudaMoeTopkSoftmax<Runtime<Device::Type::kMetax>> {
 public:
  using CudaMoeTopkSoftmax<Runtime<Device::Type::kMetax>>::CudaMoeTopkSoftmax;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_METAX_MOE_TOPK_SOFTMAX_KERNEL_H_