#ifndef INFINI_OPS_QY_ARGMAX_KERNEL_H_
#define INFINI_OPS_QY_ARGMAX_KERNEL_H_

#include "native/cuda/ops/argmax/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
class Operator<Argmax, Device::Type::kQy>
    : public CudaArgmax<Runtime<Device::Type::kQy>> {
 public:
  using CudaArgmax<Runtime<Device::Type::kQy>>::CudaArgmax;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_QY_ARGMAX_KERNEL_H_
