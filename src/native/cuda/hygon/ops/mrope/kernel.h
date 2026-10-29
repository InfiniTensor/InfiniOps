#pragma once
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/mrope/kernel.h"
namespace infini::ops {
template <>
class Operator<Mrope, Device::Type::kHygon>
    : public CudaMrope<Runtime<Device::Type::kHygon>> {
 public:
  using CudaMrope<Runtime<Device::Type::kHygon>>::CudaMrope;
};
}  // namespace infini::ops
