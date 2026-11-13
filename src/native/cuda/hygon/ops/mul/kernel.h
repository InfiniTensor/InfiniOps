#pragma once

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/mul/kernel.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kHygon>
    : public CudaMul<Runtime<Device::Type::kHygon>> {
 public:
  using CudaMul<Runtime<Device::Type::kHygon>>::CudaMul;
};

}  // namespace infini::ops
