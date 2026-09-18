#ifndef INFINI_OPS_METAX_MOE_ALIGN_KERNEL_H_
#define INFINI_OPS_METAX_MOE_ALIGN_KERNEL_H_

#include <utility>

#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/moe_align/kernel.h"

namespace infini::ops {

template <>
class Operator<MoeAlign, Device::Type::kMetax>
    : public CudaMoeAlign<Runtime<Device::Type::kMetax>> {
 public:
  using CudaMoeAlign<Runtime<Device::Type::kMetax>>::CudaMoeAlign;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_METAX_MOE_ALIGN_KERNEL_H_