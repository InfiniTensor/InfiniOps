#ifndef INFINI_OPS_HYGON_GEMM_CUBLAS_H_
#define INFINI_OPS_HYGON_GEMM_CUBLAS_H_

#include <utility>

// clang-format off
#include "cublas_v2.h"
// clang-format on

#include "cuda/gemm/blas.h"

namespace infini::ops {

namespace gemm {

struct HygonBackend {
  using blasHandle_t = cublasHandle_t;

  using stream_t = cudaStream_t;

  static constexpr auto BLAS_OP_N = CUBLAS_OP_N;

  static constexpr auto BLAS_OP_T = CUBLAS_OP_T;

  static constexpr auto R_16F = CUDA_R_16F;

  static constexpr auto R_16BF = CUDA_R_16BF;

  static constexpr auto R_32F = CUDA_R_32F;

  static constexpr auto BLAS_COMPUTE_32F = CUBLAS_COMPUTE_32F;

  // DTK exposes the TF32 enum for compatibility, but BW/GFX9-class Hygon
  // devices do not provide a working TF32 GEMM fast path.
  static constexpr auto BLAS_COMPUTE_32F_FAST_TF32 = CUBLAS_COMPUTE_32F;

  static constexpr auto BLAS_GEMM_DEFAULT = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

  static constexpr auto blasCreate = cublasCreate;

  static constexpr auto blasSetStream = cublasSetStream;

  static constexpr auto blasDestroy = cublasDestroy;

  static constexpr auto blasGemmEx = [](auto&&... args) {
    return cublasGemmEx(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto blasGemmStridedBatchedEx = [](auto&&... args) {
    return cublasGemmStridedBatchedEx(std::forward<decltype(args)>(args)...);
  };

  static auto GetDataType(DataType dtype) {
    if (dtype == DataType::kFloat16) return R_16F;
    if (dtype == DataType::kBFloat16) return R_16BF;
    return R_32F;
  }

  static auto GetComputeType(DataType dtype) {
    if (dtype == DataType::kFloat16 || dtype == DataType::kBFloat16)
      return BLAS_COMPUTE_32F;
    return BLAS_COMPUTE_32F_FAST_TF32;
  }
};

}  // namespace gemm

template <>
class Operator<Gemm, Device::Type::kHygon> : public Blas<gemm::HygonBackend> {
 public:
  using Blas<gemm::HygonBackend>::Blas;

  void operator()(const Tensor a, const Tensor b, std::optional<float> alpha,
                  std::optional<float> beta, std::optional<int> trans_a,
                  std::optional<int> trans_b, Tensor c) const override {
    if (this->handle_ == nullptr) {
      gemm::HygonBackend::blasCreate(&this->handle_);
    }

    gemm::HygonBackend::blasSetStream(
        this->handle_, static_cast<gemm::HygonBackend::stream_t>(this->stream_));

    const auto& alpha_value{alpha.value_or(this->alpha_)};
    const auto& beta_value{beta.value_or(this->beta_)};

    const auto& trans_a_value{trans_a.value_or(this->trans_a_)};
    const auto& trans_b_value{trans_b.value_or(this->trans_b_)};
    auto op_a{this->GetOpA(trans_a_value, trans_b_value)};
    auto op_b{this->GetOpB(trans_a_value, trans_b_value)};
    const void* alpha_ptr{this->GetAlphaPtr(alpha_value, c.dtype())};
    const void* beta_ptr{this->GetBetaPtr(beta_value, c.dtype())};

    if (this->batch_count_ == 1) {
      gemm::HygonBackend::blasGemmEx(
          this->handle_, op_a, op_b,
          this->swap_a_and_b_ ? this->n_ : this->m_,
          this->swap_a_and_b_ ? this->m_ : this->n_, this->k_, alpha_ptr,
          this->swap_a_and_b_ ? b.data() : a.data(),
          gemm::HygonBackend::GetDataType(this->swap_a_and_b_ ? b.dtype()
                                                               : a.dtype()),
          this->swap_a_and_b_ ? this->ldb_ : this->lda_,
          this->swap_a_and_b_ ? a.data() : b.data(),
          gemm::HygonBackend::GetDataType(this->swap_a_and_b_ ? a.dtype()
                                                               : b.dtype()),
          this->swap_a_and_b_ ? this->lda_ : this->ldb_, beta_ptr, c.data(),
          gemm::HygonBackend::GetDataType(c.dtype()), this->ldc_,
          gemm::HygonBackend::GetComputeType(c.dtype()),
          gemm::HygonBackend::BLAS_GEMM_DEFAULT);
      gemm::HygonBackend::blasDestroy(this->handle_);
      this->handle_ = nullptr;
      return;
    }

    Blas<gemm::HygonBackend>::operator()(a, b, alpha, beta, trans_a, trans_b,
                                         c);
    gemm::HygonBackend::blasDestroy(this->handle_);
    this->handle_ = nullptr;
  }
};

}  // namespace infini::ops

#endif
