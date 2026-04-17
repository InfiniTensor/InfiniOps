#ifndef INFINI_OPS_MOORE_CAST_KERNEL_H_
#define INFINI_OPS_MOORE_CAST_KERNEL_H_

#include <utility>

// clang-format off
#include "moore/polyfills.cuh"
// clang-format on

#include "cuda/cast/kernel.h"
#include "moore/caster.cuh"
#include "moore/polyfills.cuh"
#include "moore/runtime_.h"

namespace infini::ops {

template <>
class Operator<Cast, Device::Type::kMoore>
    : public CudaCast<Runtime<Device::Type::kMoore>> {
 public:
  using CudaCast<Runtime<Device::Type::kMoore>>::CudaCast;
};

}  // namespace infini::ops

#endif
