#ifndef INFINI_OPS_NVIDIA_SWIGLU_DSL_H_
#define INFINI_OPS_NVIDIA_SWIGLU_DSL_H_

#include <utility>

#include "impl.h"
#include "nvidia/swiglu/registry.h"

#include "cuda/swiglu/dsl.h"
#include "nvidia/caster.cuh"
#include "nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<Swiglu, Device::Type::kNvidia, Impl::kDsl>
    : public DslCudaSwiglu<Runtime<Device::Type::kNvidia>> {
 public:
  using DslCudaSwiglu<Runtime<Device::Type::kNvidia>>::DslCudaSwiglu;
};

}  // namespace infini::ops

#endif
