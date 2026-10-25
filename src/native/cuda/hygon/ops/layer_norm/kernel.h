#pragma once
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/layer_norm/kernel.h"
namespace infini::ops {
template <>
class Operator<LayerNorm, Device::Type::kHygon, 0>
    : public CudaLayerNorm<Device::Type::kHygon> {
 public:
  using CudaLayerNorm<Device::Type::kHygon>::CudaLayerNorm;
};
}  // namespace infini::ops
