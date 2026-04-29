#ifndef INFINI_OPS_MOORE_SWIGLU_KERNEL_H_
#define INFINI_OPS_MOORE_SWIGLU_KERNEL_H_

#include <utility>

// clang-format off
#include "cuda/moore/polyfills.cuh"
// clang-format on

#include "cuda/moore/caster.cuh"
#include "cuda/moore/polyfills.cuh"
#include "cuda/moore/runtime_.h"
#include "cuda/swiglu/kernel.h"

namespace infini::ops {

template <>
class Operator<Swiglu, Device::Type::kMoore>
    : public CudaSwiglu<Runtime<Device::Type::kMoore>> {
 public:
  using CudaSwiglu<Runtime<Device::Type::kMoore>>::CudaSwiglu;
};

}  // namespace infini::ops

#endif
