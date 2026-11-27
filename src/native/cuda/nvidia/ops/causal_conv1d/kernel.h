#ifndef INFINI_OPS_NVIDIA_CAUSAL_CONV1D_KERNEL_H_
#define INFINI_OPS_NVIDIA_CAUSAL_CONV1D_KERNEL_H_

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/causal_conv1d/kernel.h"

namespace infini::ops {

template <>
class Operator<CausalConv1d, Device::Type::kNvidia>
    : public CudaCausalConv1d<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaCausalConv1d<Runtime<Device::Type::kNvidia>>::CudaCausalConv1d;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_CAUSAL_CONV1D_KERNEL_H_
