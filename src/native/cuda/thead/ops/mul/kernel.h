#ifndef INFINI_OPS_THEAD_MUL_KERNEL_H_
#define INFINI_OPS_THEAD_MUL_KERNEL_H_

#include <utility>

#include "native/cuda/ops/mul/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kThead>
    : public CudaMul<Runtime<Device::Type::kThead>> {
 public:
  using CudaMul<Runtime<Device::Type::kThead>>::CudaMul;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_THEAD_MUL_KERNEL_H_
