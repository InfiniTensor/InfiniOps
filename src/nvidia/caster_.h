#ifndef INFINI_OPS_NVIDIA_CASTER__H_
#define INFINI_OPS_NVIDIA_CASTER__H_

#include "cuda/caster.cuh"
#include "nvidia/data_type_.h"

namespace infini::ops {

template <>
struct Caster<Device::Type::kNvidia> : CudaCasterImpl<Device::Type::kNvidia> {};

}  // namespace infini::ops

#endif
