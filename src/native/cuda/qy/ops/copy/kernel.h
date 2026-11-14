#ifndef INFINI_OPS_QY_COPY_KERNEL_H_
#define INFINI_OPS_QY_COPY_KERNEL_H_

#include "native/cuda/ops/copy/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
class Operator<Copy, Device::Type::kQy>
    : public CudaCopy<Runtime<Device::Type::kQy>> {
 public:
  using CudaCopy<Runtime<Device::Type::kQy>>::CudaCopy;
};

}  // namespace infini::ops

#endif
