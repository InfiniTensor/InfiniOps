#ifndef INFINI_OPS_THEAD_FILL_KERNEL_H_
#define INFINI_OPS_THEAD_FILL_KERNEL_H_

#include <utility>

#include "native/cuda/ops/fill/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<Fill, Device::Type::kThead>
    : public CudaFill<Runtime<Device::Type::kThead>> {
 public:
  using CudaFill<Runtime<Device::Type::kThead>>::CudaFill;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_THEAD_FILL_KERNEL_H_
