#ifndef INFINI_OPS_HYGON_CLOVER_live_mask_KERNEL_H_
#define INFINI_OPS_HYGON_CLOVER_live_mask_KERNEL_H_

#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/clover_live_mask/kernel.h"

namespace infini::ops {

template <>
class Operator<CloverLiveMask, Device::Type::kHygon>
    : public CudaCloverLiveMask<Runtime<Device::Type::kHygon>> {
 public:
  using CudaCloverLiveMask<Runtime<Device::Type::kHygon>>::CudaCloverLiveMask;
};

}  // namespace infini::ops

#endif
