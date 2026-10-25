#pragma once
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/pad/kernel.h"
namespace infini::ops {
template <>
class Operator<Pad, Device::Type::kHygon, 0>
    : public CudaPad<Device::Type::kHygon> {
 public:
  using CudaPad<Device::Type::kHygon>::CudaPad;
};
}  // namespace infini::ops
