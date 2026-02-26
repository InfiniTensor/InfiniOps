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
      : Gemm{a, b, alpha, beta, trans_a, trans_b, c},
        swapped_a_and_b_{c_strides_[1] == 1} {
    Backend::blasCreate(&handle_);
    // TODO: Check constraints.
  }

  ~Blas() { Backend::blasDestroy(handle_); }

  Blas(const Tensor a, const Tensor b, std::optional<float> alpha,
       std::optional<float> beta, Tensor c)
      : Blas{a, b, alpha, beta, std::nullopt, std::nullopt, c} {}

  Blas(const Tensor a, const Tensor b, Tensor c)
      : Blas{a, b, std::nullopt, std::nullopt, std::nullopt, std::nullopt, c} {}

  void operator()(void* stream, const Tensor a, const Tensor b,
                  std::optional<float> alpha, std::optional<float> beta,
                  std::optional<int> trans_a, std::optional<int> trans_b,
                  Tensor c) const override {
    Backend::blasSetStream(handle_,
                           static_cast<typename Backend::stream_t>(stream));

    const auto& alpha_value{alpha.value_or(alpha_)};
    const auto& beta_value{beta.value_or(beta_)};

    const auto& trans_a_value{trans_a.value_or(trans_a_)};
    const auto& trans_b_value{trans_b.value_or(trans_b_)};
    auto op_a{GetOpA(trans_a_value, trans_b_value)};
    auto op_b{GetOpB(trans_a_value, trans_b_value)};

    Backend::blasGemmEx(
        handle_, op_a, op_b, swapped_a_and_b_ ? n_ : m_,
        swapped_a_and_b_ ? m_ : n_, k_, &alpha_value,
        swapped_a_and_b_ ? b.data() : a.data(), Backend::R_32F,
        swapped_a_and_b_ ? ldb_ : lda_, swapped_a_and_b_ ? a.data() : b.data(),
        Backend::R_32F, swapped_a_and_b_ ? lda_ : ldb_, &beta_value, c.data(),
        Backend::R_32F, ldc_, Backend::BLAS_COMPUTE_32F_FAST_TF32,
        Backend::BLAS_GEMM_DEFAULT);
  }

 private:
  auto GetOpA(int trans_a, int trans_b) const {
    if (swapped_a_and_b_) {
      return ((b_strides_[1] == 1) == trans_b) ? Backend::BLAS_OP_T
                                               : Backend::BLAS_OP_N;
    }
    return ((a_strides_[1] == 1) != trans_a) ? Backend::BLAS_OP_T
                                             : Backend::BLAS_OP_N;
  }

  auto GetOpB(int trans_a, int trans_b) const {
    if (swapped_a_and_b_) {
      return ((a_strides_[1] == 1) == trans_a) ? Backend::BLAS_OP_T
                                               : Backend::BLAS_OP_N;
    }
    return ((b_strides_[1] == 1) != trans_b) ? Backend::BLAS_OP_T
                                             : Backend::BLAS_OP_N;
  }

  bool swapped_a_and_b_{false};
  typename Backend::blasHandle_t handle_;
};

}  // namespace infini::ops

#endif
