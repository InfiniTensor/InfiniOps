#ifndef INFINI_OPS_QY_SIGMOID_KERNEL_H_
#define INFINI_OPS_QY_SIGMOID_KERNEL_H_

#include <utility>

#include "native/cuda/ops/sigmoid/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
class Operator<Sigmoid, Device::Type::kQy>
    : public CudaSigmoid<Runtime<Device::Type::kQy>> {
 public:
  using CudaSigmoid<Runtime<Device::Type::kQy>>::CudaSigmoid;
};

}  // namespace infini::ops

#endif
