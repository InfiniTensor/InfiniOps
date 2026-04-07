#ifndef INFINI_OPS_ILUVATAR_GEMM_CUBLAS_H_
#define INFINI_OPS_ILUVATAR_GEMM_CUBLAS_H_

#include "cuda/gemm/blas.h"
#include "iluvatar/blas.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kIluvatar>
    : public BlasGemm<Blas<Device::Type::kIluvatar>> {
 public:
  using BlasGemm<Blas<Device::Type::kIluvatar>>::BlasGemm;
};

}  // namespace infini::ops

#endif
