#pragma once
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/causal_conv1d/kernel.h"
namespace infini::ops {
template <>
class Operator<CausalConv1d, Device::Type::kNvidia, 0>
    : public CudaCausalConv1d<Device::Type::kNvidia> {
 public:
  using CudaCausalConv1d<Device::Type::kNvidia>::CudaCausalConv1d;
};
}  // namespace infini::ops
