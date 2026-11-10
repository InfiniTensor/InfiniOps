#ifndef INFINI_OPS_MOORE_MATMUL_KERNEL_H_
#define INFINI_OPS_MOORE_MATMUL_KERNEL_H_

#include <utility>

// clang-format off
#include "native/cuda/moore/polyfills.cuh"
// clang-format on
#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/matmul/kernel.h"

namespace infini::ops {

template <>
class Operator<Matmul, Device::Type::kMoore>
    : public CudaMatmul<Runtime<Device::Type::kMoore>> {
 public:
  using CudaMatmul<Runtime<Device::Type::kMoore>>::CudaMatmul;
};

}  // namespace infini::ops

#endif
