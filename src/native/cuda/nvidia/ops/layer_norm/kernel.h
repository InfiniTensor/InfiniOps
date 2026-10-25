#pragma once
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/layer_norm/kernel.h"
namespace infini::ops {
template <>
class Operator<LayerNorm, Device::Type::kNvidia, 0>
    : public CudaLayerNorm<Device::Type::kNvidia> {
 public:
  using CudaLayerNorm<Device::Type::kNvidia>::CudaLayerNorm;
};
}  // namespace infini::ops
