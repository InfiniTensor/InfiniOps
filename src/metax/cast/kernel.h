#ifndef INFINI_OPS_METAX_CAST_KERNEL_H_
#define INFINI_OPS_METAX_CAST_KERNEL_H_

#include <utility>

#include "cuda/cast/kernel.h"
#include "metax/caster.cuh"
#include "metax/runtime_.h"

namespace infini::ops {

template <>
class Operator<Cast, Device::Type::kMetax>
    : public CudaCast<Runtime<Device::Type::kMetax>> {
 public:
  using CudaCast<Runtime<Device::Type::kMetax>>::CudaCast;
};

}  // namespace infini::ops

#endif
