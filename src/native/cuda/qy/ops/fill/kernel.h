#ifndef INFINI_OPS_QY_FILL_KERNEL_H_
#define INFINI_OPS_QY_FILL_KERNEL_H_

#include <utility>

#include "native/cuda/ops/fill/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
class Operator<Fill, Device::Type::kQy>
    : public CudaFill<Runtime<Device::Type::kQy>> {
 public:
  using CudaFill<Runtime<Device::Type::kQy>>::CudaFill;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_QY_FILL_KERNEL_H_
