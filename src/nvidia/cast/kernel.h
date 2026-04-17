#ifndef INFINI_OPS_NVIDIA_CAST_KERNEL_H_
#define INFINI_OPS_NVIDIA_CAST_KERNEL_H_

#include <utility>

#include "cuda/cast/kernel.h"
#include "nvidia/caster.cuh"
#include "nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<Cast, Device::Type::kNvidia>
    : public CudaCast<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaCast<Runtime<Device::Type::kNvidia>>::CudaCast;
};

}  // namespace infini::ops

#endif
