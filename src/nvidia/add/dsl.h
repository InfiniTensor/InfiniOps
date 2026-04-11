#ifndef INFINI_OPS_NVIDIA_ADD_DSL_H_
#define INFINI_OPS_NVIDIA_ADD_DSL_H_

#include <utility>

#include "impl.h"
#include "nvidia/add/registry.h"

#include "cuda/add/dsl.h"
#include "nvidia/caster.cuh"
#include "nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kNvidia, Impl::kDsl>
    : public DslCudaAdd<Runtime<Device::Type::kNvidia>> {
 public:
  using DslCudaAdd<Runtime<Device::Type::kNvidia>>::DslCudaAdd;
};

}  // namespace infini::ops

#endif
