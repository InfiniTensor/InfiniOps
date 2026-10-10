#ifndef INFINI_OPS_QY_MUL_KERNEL_H_
#define INFINI_OPS_QY_MUL_KERNEL_H_

#include <utility>

#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"
#include "native/cuda/ops/mul/kernel.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kQy>
    : public CudaMul<Runtime<Device::Type::kQy>> {
 public:
  using CudaMul<Runtime<Device::Type::kQy>>::CudaMul;
};

}  // namespace infini::ops

#endif
