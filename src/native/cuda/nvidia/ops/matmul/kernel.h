#ifndef INFINI_OPS_NVIDIA_MATMUL_KERNEL_H_
#define INFINI_OPS_NVIDIA_MATMUL_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/matmul/kernel.h"

namespace infini::ops {

template <>
class Operator<Matmul, Device::Type::kNvidia>
    : public CudaMatmul<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaMatmul<Runtime<Device::Type::kNvidia>>::CudaMatmul;
};

}  // namespace infini::ops

#endif
