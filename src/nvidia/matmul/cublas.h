#ifndef INFINI_OPS_NVIDIA_MATMUL_CUBLAS_H_
#define INFINI_OPS_NVIDIA_MATMUL_CUBLAS_H_

#include "cuda/matmul/blas.h"
#include "nvidia/blas.h"
#include "nvidia/matmul/registry.h"

namespace infini::ops {

template <>
class Operator<Matmul, Device::Type::kNvidia, 1>
    : public BlasMatmul<Blas<Device::Type::kNvidia>> {
 public:
  using BlasMatmul<Blas<Device::Type::kNvidia>>::BlasMatmul;
};

}  // namespace infini::ops

#endif
