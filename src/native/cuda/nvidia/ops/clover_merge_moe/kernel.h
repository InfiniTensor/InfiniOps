#ifndef INFINI_OPS_NVIDIA_CLOVER_merge_moe_KERNEL_H_
#define INFINI_OPS_NVIDIA_CLOVER_merge_moe_KERNEL_H_

#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/clover_merge_moe/kernel.h"

namespace infini::ops {

template <>
class Operator<CloverMergeMoe, Device::Type::kNvidia>
    : public CudaCloverMergeMoe<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaCloverMergeMoe<Runtime<Device::Type::kNvidia>>::CudaCloverMergeMoe;
};

}  // namespace infini::ops

#endif
