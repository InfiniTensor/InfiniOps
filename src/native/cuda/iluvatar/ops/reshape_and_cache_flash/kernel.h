#ifndef INFINI_OPS_ILUVATAR_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
#define INFINI_OPS_ILUVATAR_RESHAPE_AND_CACHE_FLASH_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/reshape_and_cache_flash/kernel.h"

namespace infini::ops {

template <>
class Operator<ReshapeAndCacheFlash, Device::Type::kIluvatar>
    : public CudaReshapeAndCacheFlash<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaReshapeAndCacheFlash<
      Runtime<Device::Type::kIluvatar>>::CudaReshapeAndCacheFlash;
};

}  // namespace infini::ops

#endif
