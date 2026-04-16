#ifndef INFINI_OPS_MOORE_ADD_KERNEL_H_
#define INFINI_OPS_MOORE_ADD_KERNEL_H_

#include <utility>

// clang-format off
#include "cuda/moore/polyfills.cuh"
// clang-format on

#include "cuda/add/kernel.h"
#include "cuda/moore/caster.cuh"
#include "cuda/moore/polyfills.cuh"
#include "cuda/moore/runtime_.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kMoore>
    : public CudaAdd<Runtime<Device::Type::kMoore>> {
 public:
  using CudaAdd<Runtime<Device::Type::kMoore>>::CudaAdd;
};

}  // namespace infini::ops

#endif
