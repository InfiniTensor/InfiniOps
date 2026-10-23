#ifndef INFINI_OPS_QY_GEMM_CUBLAS_H_
#define INFINI_OPS_QY_GEMM_CUBLAS_H_

#include "native/cuda/ops/gemm/blas.h"
#include "native/cuda/qy/blas.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kQy, 0>
    : public BlasGemm<Blas<Device::Type::kQy>> {
 public:
  using BlasGemm<Blas<Device::Type::kQy>>::BlasGemm;
};

}  // namespace infini::ops

#endif
