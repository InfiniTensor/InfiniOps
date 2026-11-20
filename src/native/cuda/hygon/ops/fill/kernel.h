#pragma once

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/fill/kernel.h"

namespace infini::ops {

template <>
class Operator<Fill, Device::Type::kHygon>
    : public CudaFill<Runtime<Device::Type::kHygon>> {
 public:
  using CudaFill<Runtime<Device::Type::kHygon>>::CudaFill;
};

}  // namespace infini::ops
