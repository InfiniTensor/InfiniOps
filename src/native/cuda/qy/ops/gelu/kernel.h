#ifndef INFINI_OPS_QY_GELU_KERNEL_H_
#define INFINI_OPS_QY_GELU_KERNEL_H_

#include <utility>

#include "native/cuda/ops/gelu/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
class Operator<Gelu, Device::Type::kQy>
    : public CudaGelu<Runtime<Device::Type::kQy>> {
 public:
  using CudaGelu<Runtime<Device::Type::kQy>>::CudaGelu;
};

}  // namespace infini::ops

#endif
