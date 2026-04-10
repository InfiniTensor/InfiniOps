#ifndef INFINI_OPS_MOORE_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_MOORE_ADD_RMS_NORM_KERNEL_H_

#include <utility>

// clang-format off
#include "moore/polyfills.cuh"
// clang-format on

#include "cuda/add_rms_norm/kernel.h"
#include "moore/add/kernel.h"
#include "moore/add_rms_norm/registry.h"
#include "moore/caster.cuh"
#include "moore/polyfills.cuh"
#include "moore/rms_norm/kernel.h"
#include "moore/runtime_.h"

namespace infini::ops {

template <>
class Operator<AddRmsNorm, Device::Type::kMoore, 0>
    : public CudaAddRmsNorm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaAddRmsNorm<Runtime<Device::Type::kMoore>>::CudaAddRmsNorm;
};

template <>
class Operator<AddRmsNorm, Device::Type::kMoore, 1>
    : public CudaAddRmsNormFused<Runtime<Device::Type::kMoore>> {
 public:
  using CudaAddRmsNormFused<Runtime<Device::Type::kMoore>>::CudaAddRmsNormFused;
};

}  // namespace infini::ops

#endif
