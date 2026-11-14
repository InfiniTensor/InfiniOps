#ifndef INFINI_OPS_QY_SILU_KERNEL_H_
#define INFINI_OPS_QY_SILU_KERNEL_H_

#include <utility>

#include "native/cuda/ops/silu/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
class Operator<Silu, Device::Type::kQy>
    : public CudaSilu<Runtime<Device::Type::kQy>> {
 public:
  using CudaSilu<Runtime<Device::Type::kQy>>::CudaSilu;
};

}  // namespace infini::ops

#endif
