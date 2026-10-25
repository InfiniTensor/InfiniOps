#pragma once
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/quick_gelu/kernel.h"
namespace infini::ops {
template <>
class Operator<QuickGelu, Device::Type::kNvidia, 0>
    : public CudaQuickGelu<Device::Type::kNvidia> {
 public:
  using CudaQuickGelu<Device::Type::kNvidia>::CudaQuickGelu;
};
}  // namespace infini::ops
