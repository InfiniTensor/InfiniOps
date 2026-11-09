#ifndef INFINI_OPS_ILUVATAR_GEMM_CUBLAS_H_
#define INFINI_OPS_ILUVATAR_GEMM_CUBLAS_H_

#include "native/cuda/iluvatar/blas.h"
#include "native/cuda/iluvatar/ops/gemm/gemv_policy.h"
#include "native/cuda/iluvatar/ops/gemm/kernel.h"
#include "native/cuda/ops/gemm/blas.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kIluvatar>
    : public BlasGemm<Blas<Device::Type::kIluvatar>> {
  using Base = BlasGemm<Blas<Device::Type::kIluvatar>>;

 public:
  using Base::Base;

  using Base::operator();

  void operator()(const Tensor a, const Tensor b, const std::optional<Tensor> c,
                  std::optional<float> alpha, std::optional<float> beta,
                  std::optional<int> trans_a, std::optional<int> trans_b,
                  Tensor y) const override {
    if (a.ndim() == 2 && b.ndim() == 2 && y.ndim() == 2 &&
        a.size(1) == b.size(0) && a.size(0) == y.size(0) &&
        b.size(1) == y.size(1) &&
        iluvatar::CanUseSmallBatchGemv(
            a.dtype() == DataType::kBFloat16 &&
                b.dtype() == DataType::kBFloat16 &&
                y.dtype() == DataType::kBFloat16,
            a.device() == b.device() && a.device() == y.device(), c.has_value(),
            alpha.value_or(alpha_), beta.value_or(beta_),
            trans_a.value_or(trans_a_), trans_b.value_or(trans_b_), a.size(0),
            a.size(1), y.size(1), a.stride(0), a.stride(1), b.stride(0),
            b.stride(1), y.stride(0), y.stride(1),
            reinterpret_cast<std::uintptr_t>(a.data()),
            reinterpret_cast<std::uintptr_t>(b.data()))) {
      auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : 0);
      iluvatar::LaunchSmallBatchGemv(a.data(), b.data(), y.data(), a.size(0),
                                     a.size(1), y.size(1), stream);
      return;
    }
    Base::operator()(a, b, c, alpha, beta, trans_a, trans_b, y);
  }
};

}  // namespace infini::ops

#endif
