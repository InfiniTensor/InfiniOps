#ifndef INFINI_OPS_NVIDIA_GEMM_CUBLAS_H_
#define INFINI_OPS_NVIDIA_GEMM_CUBLAS_H_

#include "cuda/gemm/blas.h"
#include "nvidia/blas.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kNvidia, 0>
    : public BlasGemm<Blas<Device::Type::kNvidia>> {
 public:
  using BlasGemm<Blas<Device::Type::kNvidia>>::BlasGemm;
};

}  // namespace infini::ops

#endif
