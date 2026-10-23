#ifndef INFINI_OPS_QY_ADD_KERNEL_H_
#define INFINI_OPS_QY_ADD_KERNEL_H_

#include "native/cuda/ops/add/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kQy>
    : public CudaAdd<Runtime<Device::Type::kQy>> {
 public:
  using CudaAdd<Runtime<Device::Type::kQy>>::CudaAdd;
};

}  // namespace infini::ops

#endif
