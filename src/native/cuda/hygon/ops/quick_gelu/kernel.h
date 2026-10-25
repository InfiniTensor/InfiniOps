#pragma once
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/quick_gelu/kernel.h"
namespace infini::ops {
template <>
class Operator<QuickGelu, Device::Type::kHygon, 0>
    : public CudaQuickGelu<Device::Type::kHygon> {
 public:
  using CudaQuickGelu<Device::Type::kHygon>::CudaQuickGelu;
};
}  // namespace infini::ops
