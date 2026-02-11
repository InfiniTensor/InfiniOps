#ifndef INFINI_OPS_NVIDIA_GEMM_CUBLAS_H_
#define INFINI_OPS_NVIDIA_GEMM_CUBLAS_H_

#include <utility>

// clang-format off
#include "cublas_v2.h"
// clang-format on

#include "base/gemm.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kNvidia> : public Gemm {
 public:
  Operator(const Tensor a, const Tensor b, std::optional<float> alpha,
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

  Operator(const Tensor a, const Tensor b, Tensor c)
      : Operator{a, b, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                 c} {}

  void operator()(void* stream, const Tensor a, const Tensor b,
                  std::optional<float> alpha, std::optional<float> beta,
                  std::optional<int> trans_a, std::optional<int> trans_b,
                  Tensor c) const override {
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetStream(handle, static_cast<cudaStream_t>(stream));

    const auto& alpha_value{alpha.value_or(alpha_)};
    const auto& beta_value{beta.value_or(beta_)};
    const auto& trans_a_value{alpha.value_or(trans_a_)};
    const auto& trans_b_value{beta.value_or(trans_b_)};

    // TODO: Add support for more data types.
    assert(a_type_ == kFloat32 && b_type_ == kFloat32 && c_type_ == kFloat32 &&
           "`operator()` not implemented for this data type");

    cublasGemmEx(handle, trans_a_value ? CUBLAS_OP_T : CUBLAS_OP_N,
                 trans_b_value ? CUBLAS_OP_T : CUBLAS_OP_N, m_, n_, k_,
                 &alpha_value, b.data(), CUDA_R_32F, lda_, a.data(), CUDA_R_32F,
                 ldb_, &beta_value, c.data(), CUDA_R_32F, ldc_,
                 CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);

    cublasDestroy(handle);
  }

 private:
  Tensor::Stride lda_{0};

  Tensor::Stride ldb_{0};

  Tensor::Stride ldc_{0};
};

}  // namespace infini::ops

#endif
