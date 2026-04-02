#ifndef INFINI_OPS_METAX_CASTER__H_
#define INFINI_OPS_METAX_CASTER__H_

#include "cuda/caster.cuh"
#include "metax/data_type_.h"

namespace infini::ops {

template <>
struct Caster<Device::Type::kMetax> : CudaCasterImpl<Device::Type::kMetax> {};

}  // namespace infini::ops

#endif
