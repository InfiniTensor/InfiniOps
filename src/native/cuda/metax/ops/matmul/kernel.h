#ifndef INFINI_OPS_METAX_MATMUL_KERNEL_H_
#define INFINI_OPS_METAX_MATMUL_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/matmul/kernel.h"

namespace infini::ops {

template <>
class Operator<Matmul, Device::Type::kMetax>
    : public CudaMatmul<Runtime<Device::Type::kMetax>> {
 public:
  using CudaMatmul<Runtime<Device::Type::kMetax>>::CudaMatmul;
};

}  // namespace infini::ops

#endif
