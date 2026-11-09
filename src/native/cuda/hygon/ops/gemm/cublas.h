#ifndef INFINI_OPS_HYGON_GEMM_CUBLAS_H_
#define INFINI_OPS_HYGON_GEMM_CUBLAS_H_

#include "native/cuda/hygon/blas.h"
#include "native/cuda/ops/gemm/blas.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kHygon>
    : public BlasGemm<Blas<Device::Type::kHygon>> {
 public:
  using BlasGemm<Blas<Device::Type::kHygon>>::BlasGemm;
};

}  // namespace infini::ops

#endif
