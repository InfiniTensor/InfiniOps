#ifndef INFINI_OPS_Nvidia_CLOVER_BLOCK_CONV1D_KERNEL_H_
#define INFINI_OPS_Nvidia_CLOVER_BLOCK_CONV1D_KERNEL_H_

#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/clover_block_conv1d/kernel.h"

namespace infini::ops {

template <>
class Operator<CloverBlockConv1d, Device::Type::kNvidia>
    : public CudaCloverBlockConv1d<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaCloverBlockConv1d<
      Runtime<Device::Type::kNvidia>>::CudaCloverBlockConv1d;
};

}  // namespace infini::ops

#endif
