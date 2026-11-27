#ifndef INFINI_OPS_HYGON_CLOVER_BLOCK_CONV1D_KERNEL_H_
#define INFINI_OPS_HYGON_CLOVER_BLOCK_CONV1D_KERNEL_H_

#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/clover_block_conv1d/kernel.h"

namespace infini::ops {

template <>
class Operator<CloverBlockConv1d, Device::Type::kHygon>
    : public CudaCloverBlockConv1d<Runtime<Device::Type::kHygon>> {
 public:
  using CudaCloverBlockConv1d<
      Runtime<Device::Type::kHygon>>::CudaCloverBlockConv1d;
};

}  // namespace infini::ops

#endif
