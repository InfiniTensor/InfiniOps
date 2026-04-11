#ifndef INFINI_OPS_NVIDIA_RMS_NORM_DSL_H_
#define INFINI_OPS_NVIDIA_RMS_NORM_DSL_H_

#include <utility>

#include "impl.h"
#include "nvidia/rms_norm/registry.h"

#include "cuda/rms_norm/dsl.h"
#include "nvidia/caster.cuh"
#include "nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kNvidia, Impl::kDsl>
    : public DslCudaRmsNorm<Runtime<Device::Type::kNvidia>> {
 public:
  using DslCudaRmsNorm<Runtime<Device::Type::kNvidia>>::DslCudaRmsNorm;
};

}  // namespace infini::ops

#endif
