#ifndef INFINI_OPS_NVIDIA_CLOVER_mask_routes_KERNEL_H_
#define INFINI_OPS_NVIDIA_CLOVER_mask_routes_KERNEL_H_

#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/clover_mask_routes/kernel.h"

namespace infini::ops {

template <>
class Operator<CloverMaskRoutes, Device::Type::kNvidia>
    : public CudaCloverMaskRoutes<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaCloverMaskRoutes<Runtime<Device::Type::kNvidia>>::CudaCloverMaskRoutes;
};

}  // namespace infini::ops

#endif
