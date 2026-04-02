#ifndef INFINI_OPS_MOORE_CASTER__H_
#define INFINI_OPS_MOORE_CASTER__H_

#include "cuda/caster.cuh"
#include "moore/data_type_.h"

namespace infini::ops {

template <>
struct Caster<Device::Type::kMoore> : CudaCasterImpl<Device::Type::kMoore> {};

}  // namespace infini::ops

#endif
