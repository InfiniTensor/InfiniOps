#pragma once
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/causal_conv1d/kernel.h"
namespace infini::ops {
template <>
class Operator<CausalConv1d, Device::Type::kHygon, 0>
    : public CudaCausalConv1d<Device::Type::kHygon> {
 public:
  using CudaCausalConv1d<Device::Type::kHygon>::CudaCausalConv1d;
};
}  // namespace infini::ops
