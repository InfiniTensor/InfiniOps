#ifndef INFINI_OPS_HYGON_GEMM_CUBLAS_H_
#define INFINI_OPS_HYGON_GEMM_CUBLAS_H_

#include "native/cuda/hygon/blas.h"
#include "native/cuda/ops/gemm/blas.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kHygon>
    : public BlasGemm<Blas<Device::Type::kHygon>> {
 public:
  using Base = BlasGemm<Blas<Device::Type::kHygon>>;
  Operator(Tensor a, Tensor b, const std::optional<Tensor> c,
           std::optional<float> alpha, std::optional<float> beta,
           std::optional<int> trans_a, std::optional<int> trans_b, Tensor y)
      : Base(a,b,c,alpha,beta,trans_a,trans_b,y) {
    // Make runs before stream capture; initialize the rank's BLAS handle here.
    (void)this->GetHandle();
  }
  Operator(Tensor a, Tensor b, Tensor y)
      : Operator(a,b,std::nullopt,std::nullopt,std::nullopt,std::nullopt,std::nullopt,y) {}
  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif
