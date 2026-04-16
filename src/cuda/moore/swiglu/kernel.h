#ifndef INFINI_OPS_MOORE_SWIGLU_KERNEL_H_
#define INFINI_OPS_MOORE_SWIGLU_KERNEL_H_

#include <utility>

// clang-format off
#include "moore/polyfills.cuh"
// clang-format on

#include "cuda/swiglu/kernel.h"
#include "moore/caster.cuh"
#include "moore/polyfills.cuh"
#include "moore/runtime_.h"

namespace infini::ops {

template <>
class Operator<Swiglu, Device::Type::kMoore>
    : public CudaSwiglu<Runtime<Device::Type::kMoore>> {
 public:
  using CudaSwiglu<Runtime<Device::Type::kMoore>>::CudaSwiglu;
};

}  // namespace infini::ops

#endif
