#ifndef INFINI_OPS_CUDA_GEMM_BLAS_H_
#define INFINI_OPS_CUDA_GEMM_BLAS_H_

#include <utility>

#include "base/gemm.h"

namespace infini::ops {

template <typename Backend>
class Blas : public Gemm {
 public:
  Blas(const Tensor a, const Tensor b, std::optional<float> alpha,
       std::optional<float> beta, std::optional<int> trans_a,
       std::optional<int> trans_b, Tensor c)
      : Gemm{a.stride(0) == 1 ? a : b.T(),
             a.stride(0) == 1 ? b : a.T(),
             alpha,
             beta,
             trans_a,
             trans_b,
             a.stride(0) == 1 ? c : c.T()},
        lda_{a_strides_[1]},
        ldb_{b_strides_[1]},
        ldc_{c_strides_[1]} {
    // TODO: Check constraints.
  }

  Blas(const Tensor a, const Tensor b, Tensor c)
      : Blas{a, b, std::nullopt, std::nullopt, std::nullopt, std::nullopt, c} {}

  void operator()(void* stream, const Tensor a, const Tensor b,
                  std::optional<float> alpha, std::optional<float> beta,
                  std::optional<int> trans_a, std::optional<int> trans_b,
                  Tensor c) const override {
    typename Backend::blasHandle_t handle;
    Backend::blasCreate(&handle);

    Backend::blasSetStream(handle,
                           static_cast<typename Backend::stream_t>(stream));

    const auto& alpha_value{alpha.value_or(alpha_)};
    const auto& beta_value{beta.value_or(beta_)};
    const auto& trans_a_value{alpha.value_or(trans_a_)};
    const auto& trans_b_value{beta.value_or(trans_b_)};

    assert(a_type_ == kFloat32 && b_type_ == kFloat32 && c_type_ == kFloat32 &&
           "`operator()` not implemented for this data type");

    Backend::blasGemmEx(handle, trans_a_value, trans_b_value, m_, n_, k_,
                        &alpha_value, b.data(), lda_, a.data(), ldb_,
                        &beta_value, c.data(), ldc_);

    Backend::blasDestroy(handle);
  }

 private:
  Tensor::Stride lda_{0};
  Tensor::Stride ldb_{0};
  Tensor::Stride ldc_{0};
};

}  // namespace infini::ops

#endif
