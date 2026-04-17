#ifndef INFINI_OPS_NVIDIA_CAT_KERNEL_H_
#define INFINI_OPS_NVIDIA_CAT_KERNEL_H_

#include <utility>

#include "cuda/cat/kernel.h"
#include "nvidia/caster.cuh"
#include "nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<Cat, Device::Type::kNvidia>
    : public CudaCat<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaCat<Runtime<Device::Type::kNvidia>>::CudaCat;
};

}  // namespace infini::ops

#endif
