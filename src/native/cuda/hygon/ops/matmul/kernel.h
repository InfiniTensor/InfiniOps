#ifndef INFINI_OPS_HYGON_MATMUL_KERNEL_H_
#define INFINI_OPS_HYGON_MATMUL_KERNEL_H_

#include <utility>

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/matmul/kernel.h"

namespace infini::ops {

template <>
class Operator<Matmul, Device::Type::kHygon>
    : public CudaMatmul<Runtime<Device::Type::kHygon>> {
 public:
  using CudaMatmul<Runtime<Device::Type::kHygon>>::CudaMatmul;
};

}  // namespace infini::ops

#endif
