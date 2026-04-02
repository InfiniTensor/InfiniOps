#ifndef INFINI_OPS_ILUVATAR_CASTER__H_
#define INFINI_OPS_ILUVATAR_CASTER__H_

#include "cuda/caster.cuh"
#include "iluvatar/data_type_.h"

namespace infini::ops {

template <>
struct Caster<Device::Type::kIluvatar>
    : CudaCasterImpl<Device::Type::kIluvatar> {};

}  // namespace infini::ops

#endif
