#ifndef INFINI_OPS_HYGON_CLOVER_merge_moe_KERNEL_H_
#define INFINI_OPS_HYGON_CLOVER_merge_moe_KERNEL_H_

#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/clover_merge_moe/kernel.h"

namespace infini::ops {

template <>
class Operator<CloverMergeMoe, Device::Type::kHYGON>
    : public CudaCloverMergeMoe<Runtime<Device::Type::kHYGON>> {
 public:
  using CudaCloverMergeMoe<Runtime<Device::Type::kHYGON>>::CudaCloverMergeMoe;
};

}  // namespace infini::ops

#endif
