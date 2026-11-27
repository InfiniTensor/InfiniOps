#ifndef INFINI_OPS_HYGON_CLOVER_mask_routes_KERNEL_H_
#define INFINI_OPS_HYGON_CLOVER_mask_routes_KERNEL_H_

#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/clover_mask_routes/kernel.h"

namespace infini::ops {

template <>
class Operator<CloverMaskRoutes, Device::Type::kHygon>
    : public CudaCloverMaskRoutes<Runtime<Device::Type::kHygon>> {
 public:
  using CudaCloverMaskRoutes<
      Runtime<Device::Type::kHygon>>::CudaCloverMaskRoutes;
};

}  // namespace infini::ops

#endif
