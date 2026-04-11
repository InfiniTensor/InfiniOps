#ifndef INFINI_OPS_NVIDIA_LINEAR_KERNEL_H_
#define INFINI_OPS_NVIDIA_LINEAR_KERNEL_H_

#include "cuda/linear/kernel.h"
#include "nvidia/blas.h"
#include "nvidia/caster.cuh"
#include "nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<Linear, Device::Type::kNvidia>
    : public CudaLinear<Blas<Device::Type::kNvidia>> {
 public:
  using CudaLinear<Blas<Device::Type::kNvidia>>::CudaLinear;
};

}  // namespace infini::ops

#endif
