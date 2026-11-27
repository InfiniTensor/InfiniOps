#ifndef INFINI_OPS_HYGON_CAUSAL_CONV1D_KERNEL_H_
#define INFINI_OPS_HYGON_CAUSAL_CONV1D_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/causal_conv1d/kernel.h"

namespace infini::ops {

template <>
class Operator<CausalConv1d, Device::Type::kHygon>
    : public CudaCausalConv1d<Runtime<Device::Type::kHygon>> {
 public:
  using CudaCausalConv1d<Runtime<Device::Type::kHygon>>::CudaCausalConv1d;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_CAUSAL_CONV1D_KERNEL_H_
