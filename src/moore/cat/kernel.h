#ifndef INFINI_OPS_MOORE_CAT_KERNEL_H_
#define INFINI_OPS_MOORE_CAT_KERNEL_H_

#include <utility>

// clang-format off
#include "moore/polyfills.cuh"
// clang-format on

#include "cuda/cat/kernel.h"
#include "moore/caster.cuh"
#include "moore/polyfills.cuh"
#include "moore/runtime_.h"

namespace infini::ops {

template <>
class Operator<Cat, Device::Type::kMoore>
    : public CudaCat<Runtime<Device::Type::kMoore>> {
 public:
  using CudaCat<Runtime<Device::Type::kMoore>>::CudaCat;
};

}  // namespace infini::ops

#endif
