#pragma once
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/pad/kernel.h"
namespace infini::ops {
template <>
class Operator<Pad, Device::Type::kNvidia, 0>
    : public CudaPad<Device::Type::kNvidia> {
 public:
  using CudaPad<Device::Type::kNvidia>::CudaPad;
};
}  // namespace infini::ops
