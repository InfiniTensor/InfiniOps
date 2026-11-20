#ifndef INFINI_OPS_NVIDIA_CLOVER_live_mask_KERNEL_H_
#define INFINI_OPS_NVIDIA_CLOVER_live_mask_KERNEL_H_

#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/clover_live_mask/kernel.h"

namespace infini::ops {

template <>
class Operator<CloverLiveMask, Device::Type::kNvidia>
    : public CudaCloverLiveMask<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaCloverLiveMask<Runtime<Device::Type::kNvidia>>::CudaCloverLiveMask;
};

}  // namespace infini::ops

#endif
