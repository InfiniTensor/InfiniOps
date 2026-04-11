#ifndef INFINI_OPS_NVIDIA_MUL_DSL_H_
#define INFINI_OPS_NVIDIA_MUL_DSL_H_

#include <utility>

#include "impl.h"
#include "nvidia/mul/registry.h"

#include "cuda/mul/dsl.h"
#include "nvidia/caster.cuh"
#include "nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kNvidia, Impl::kDsl>
    : public DslCudaMul<Runtime<Device::Type::kNvidia>> {
 public:
  using DslCudaMul<Runtime<Device::Type::kNvidia>>::DslCudaMul;
};

}  // namespace infini::ops

#endif
