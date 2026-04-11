#ifndef INFINI_OPS_NVIDIA_CAST_DSL_H_
#define INFINI_OPS_NVIDIA_CAST_DSL_H_

#include <utility>

#include "impl.h"
#include "nvidia/cast/registry.h"

#include "cuda/cast/dsl.h"
#include "nvidia/caster.cuh"
#include "nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<Cast, Device::Type::kNvidia, Impl::kDsl>
    : public DslCudaCast<Runtime<Device::Type::kNvidia>> {
 public:
  using DslCudaCast<Runtime<Device::Type::kNvidia>>::DslCudaCast;
};

}  // namespace infini::ops

#endif
