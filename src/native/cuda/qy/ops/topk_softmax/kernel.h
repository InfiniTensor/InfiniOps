#ifndef INFINI_OPS_QY_TOPK_SOFTMAX_KERNEL_H_
#define INFINI_OPS_QY_TOPK_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/ops/topk_softmax/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
class Operator<TopkSoftmax, Device::Type::kQy>
    : public CudaTopkSoftmax<Runtime<Device::Type::kQy>> {
 public:
  using CudaTopkSoftmax<Runtime<Device::Type::kQy>>::CudaTopkSoftmax;
};

}  // namespace infini::ops

#endif
