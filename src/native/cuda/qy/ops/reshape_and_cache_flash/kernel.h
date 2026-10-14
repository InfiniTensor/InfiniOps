#ifndef INFINI_OPS_QY_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
#define INFINI_OPS_QY_RESHAPE_AND_CACHE_FLASH_KERNEL_H_

#include "native/cuda/ops/reshape_and_cache_flash/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
class Operator<ReshapeAndCacheFlash, Device::Type::kQy>
    : public CudaReshapeAndCacheFlash<Runtime<Device::Type::kQy>> {
 public:
  using CudaReshapeAndCacheFlash<
      Runtime<Device::Type::kQy>>::CudaReshapeAndCacheFlash;
};

}  // namespace infini::ops

#endif
