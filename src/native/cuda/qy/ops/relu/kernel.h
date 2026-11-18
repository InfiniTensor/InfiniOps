#ifndef INFINI_OPS_QY_RELU_KERNEL_H_
#define INFINI_OPS_QY_RELU_KERNEL_H_

#include <utility>

#include "native/cuda/ops/relu/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
class Operator<Relu, Device::Type::kQy>
    : public CudaRelu<Runtime<Device::Type::kQy>> {
 public:
  using CudaRelu<Runtime<Device::Type::kQy>>::CudaRelu;
};

}  // namespace infini::ops

#endif
