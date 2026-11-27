#ifndef INFINI_OPS_NVIDIA_CAST_KERNEL_H_
#define INFINI_OPS_NVIDIA_CAST_KERNEL_H_

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/cast/kernel.h"

namespace infini::ops {

template <>
class Operator<Cast, Device::Type::kNvidia>
    : public CudaCast<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaCast<Runtime<Device::Type::kNvidia>>::CudaCast;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_CAST_KERNEL_H_
