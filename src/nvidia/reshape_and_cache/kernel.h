#ifndef INFINI_OPS_NVIDIA_RESHAPE_AND_CACHE_KERNEL_H_
#define INFINI_OPS_NVIDIA_RESHAPE_AND_CACHE_KERNEL_H_

#include <utility>

#include "cuda/reshape_and_cache/kernel.h"
#include "nvidia/caster.cuh"
#include "nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<ReshapeAndCache, Device::Type::kNvidia>
    : public CudaReshapeAndCache<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaReshapeAndCache<Runtime<Device::Type::kNvidia>>::CudaReshapeAndCache;
};

}  // namespace infini::ops

#endif
