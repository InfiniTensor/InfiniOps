#ifndef INFINI_OPS_HYGON_CLOVER_live_mask_KERNEL_H_
#define INFINI_OPS_HYGON_CLOVER_live_mask_KERNEL_H_

#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/clover_live_mask/kernel.h"

namespace infini::ops {

template <>
class Operator<CloverLiveMask, Device::Type::kHYGON>
    : public CudaCloverLiveMask<Runtime<Device::Type::kHYGON>> {
 public:
  using CudaCloverLiveMask<Runtime<Device::Type::kHYGON>>::CudaCloverLiveMask;
};

}  // namespace infini::ops

#endif
