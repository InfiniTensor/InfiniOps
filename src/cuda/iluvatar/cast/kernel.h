#ifndef INFINI_OPS_ILUVATAR_CAST_KERNEL_H_
#define INFINI_OPS_ILUVATAR_CAST_KERNEL_H_

#include <utility>

#include "cuda/cast/kernel.h"
#include "cuda/iluvatar/caster.cuh"
#include "cuda/iluvatar/runtime_.h"

namespace infini::ops {

template <>
class Operator<Cast, Device::Type::kIluvatar>
    : public CudaCast<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaCast<Runtime<Device::Type::kIluvatar>>::CudaCast;
};

}  // namespace infini::ops

#endif
