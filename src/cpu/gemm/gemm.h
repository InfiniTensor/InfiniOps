#ifndef INFINI_OPS_CPU_GEMM_H_
#define INFINI_OPS_CPU_GEMM_H_

#include <utility>

#include "base/gemm.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kCpu> : public Gemm {
 public:
  Operator(const Tensor a, const Tensor b, std::optional<float> alpha,
           std::optional<float> beta, std::optional<int> trans_a,
           std::optional<int> trans_b, Tensor c)
      : Gemm{a, b, alpha, beta, trans_a, trans_b, c},
        lda_{a_strides_[1]},
        ldb_{b_strides_[1]},
        ldc_{c_strides_[1]} {
    // TODO: Check constraints.
  }

  Operator(const Tensor a, const Tensor b, Tensor c)
      : Operator{a, b, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                 c} {}

  void operator()(void* stream, const Tensor a, const Tensor b,
                  std::optional<float> alpha, std::optional<float> beta,
                  std::optional<int> trans_a, std::optional<int> trans_b,
                  Tensor c) const override {
    const float* A = static_cast<const float*>(a.data());
    const float* B = static_cast<const float*>(b.data());
    float* C = static_cast<float*>(c.data());

    const auto& alpha_value{alpha.value_or(alpha_)};
    const auto& beta_value{beta.value_or(beta_)};
    const auto& trans_a_value{trans_a.value_or(trans_a_)};
    const auto& trans_b_value{trans_b.value_or(trans_b_)};

    for (Tensor::Size i = 0; i < m_; ++i) {
      for (Tensor::Size j = 0; j < n_; ++j) {
        float sum = 0.0f;

        for (Tensor::Size l = 0; l < k_; ++l) {
          float a_val = trans_a_value ? A[l * m_ + i] : A[i * k_ + l];
          float b_val = trans_b_value ? B[j * k_ + l] : B[l * n_ + j];
          sum += a_val * b_val;
        }

        Tensor::Size idx = i * n_ + j;
        C[idx] = alpha_value * sum + beta_value * C[idx];
      }
    }
  }

 private:
  Tensor::Stride lda_{0};
  Tensor::Stride ldb_{0};
  Tensor::Stride ldc_{0};
};

}  // namespace infini::ops

#endif
