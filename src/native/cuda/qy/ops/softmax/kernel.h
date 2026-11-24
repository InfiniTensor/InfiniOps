#ifndef INFINI_OPS_QY_SOFTMAX_KERNEL_H_
#define INFINI_OPS_QY_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/ops/softmax/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
class Operator<Softmax, Device::Type::kQy>
    : public CudaSoftmax<Runtime<Device::Type::kQy>> {
 public:
  using CudaSoftmax<Runtime<Device::Type::kQy>>::CudaSoftmax;
};

}  // namespace infini::ops

#endif
