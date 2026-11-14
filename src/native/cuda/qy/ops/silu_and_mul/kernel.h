#ifndef INFINI_OPS_QY_SILU_AND_MUL_KERNEL_H_
#define INFINI_OPS_QY_SILU_AND_MUL_KERNEL_H_

#include "native/cuda/ops/silu_and_mul/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
class Operator<SiluAndMul, Device::Type::kQy>
    : public CudaSiluAndMul<Runtime<Device::Type::kQy>> {
 public:
  using CudaSiluAndMul<Runtime<Device::Type::kQy>>::CudaSiluAndMul;
};

}  // namespace infini::ops

#endif
