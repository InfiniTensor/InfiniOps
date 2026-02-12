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
    Backend::blasCreate(&handle);
    // TODO: Check constraints.
  }

  Blas(const Tensor a, const Tensor b, std::optional<float> alpha,
       std::optional<float> beta, Tensor c)
      : Blas{a, b, alpha, beta, std::nullopt, std::nullopt, c} {}

  Blas(const Tensor a, const Tensor b, Tensor c)
      : Blas{a, b, std::nullopt, std::nullopt, std::nullopt, std::nullopt, c} {}

  void operator()(void* stream, const Tensor a, const Tensor b,
                  std::optional<float> alpha, std::optional<float> beta,
                  std::optional<int> trans_a, std::optional<int> trans_b,
                  Tensor c) const override {
    Backend::blasSetStream(handle,
                           static_cast<typename Backend::stream_t>(stream));

    const auto& alpha_value{alpha.value_or(alpha_)};
    const auto& beta_value{beta.value_or(beta_)};
    const auto& trans_a_value{trans_a.value_or(trans_a_)};
    const auto& trans_b_value{trans_b.value_or(trans_b_)};

    assert(a_type_ == DataType::kFloat32 && b_type_ == DataType::kFloat32 &&
           c_type_ == DataType::kFloat32 &&
           "`operator()` not implemented for this data type");

    auto op_a = static_cast<decltype(Backend::BLAS_OP_T)>(
        trans_a_value ? Backend::BLAS_OP_T : Backend::BLAS_OP_N);
    auto op_b = static_cast<decltype(Backend::BLAS_OP_T)>(
        trans_b_value ? Backend::BLAS_OP_T : Backend::BLAS_OP_N);

    Backend::blasGemmEx(handle, op_a, op_b, m_, n_, k_, &alpha_value, b.data(),
                        Backend::R_32F, lda_, a.data(), Backend::R_32F, ldb_,
                        &beta_value, c.data(), Backend::R_32F, ldc_,
                        Backend::BLAS_COMPUTE_32F_FAST_TF32,
                        Backend::BLAS_GEMM_DEFAULT);

    Backend::blasDestroy(handle);
  }

 private:
  Tensor::Stride lda_{0};
  Tensor::Stride ldb_{0};
  Tensor::Stride ldc_{0};

  typename Backend::blasHandle_t handle;
};

}  // namespace infini::ops

#endif
