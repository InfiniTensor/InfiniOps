#ifndef INFINI_OPS_NVIDIA_COPY_KERNEL_H_
#define INFINI_OPS_NVIDIA_COPY_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/copy/kernel.h"

namespace infini::ops {

template <>
class Operator<Copy, Device::Type::kNvidia>
    : public CudaCopy<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaCopy<Runtime<Device::Type::kNvidia>>::CudaCopy;
};

}  // namespace infini::ops

#endif
