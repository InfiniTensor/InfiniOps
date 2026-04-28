#ifndef INFINI_OPS_HYGON_GEMM_CUBLAS_H_
#define INFINI_OPS_HYGON_GEMM_CUBLAS_H_

#include "cuda/gemm/blas.h"
#include "hygon/blas.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kHygon>
    : public BlasGemm<Blas<Device::Type::kHygon>> {
 public:
  using BlasGemm<Blas<Device::Type::kHygon>>::BlasGemm;
};

}  // namespace infini::ops

#endif
