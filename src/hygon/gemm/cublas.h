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
    const bool a_is_col_major = a.stride(-1) == 1;
    const bool b_is_col_major = b.stride(-1) == 1;
    const bool swap_a_and_b = c.stride(-1) == 1;

    auto get_op_a = [&](int trans_a_value, int trans_b_value) {
      if (swap_a_and_b) {
        return (b_is_col_major == trans_b_value) ? gemm::HygonBackend::BLAS_OP_T
                                                 : gemm::HygonBackend::BLAS_OP_N;
      }
      return (a_is_col_major != trans_a_value) ? gemm::HygonBackend::BLAS_OP_T
                                               : gemm::HygonBackend::BLAS_OP_N;
    };

    auto get_op_b = [&](int trans_a_value, int trans_b_value) {
      if (swap_a_and_b) {
        return (a_is_col_major == trans_a_value) ? gemm::HygonBackend::BLAS_OP_T
                                                 : gemm::HygonBackend::BLAS_OP_N;
      }
      return (b_is_col_major != trans_b_value) ? gemm::HygonBackend::BLAS_OP_T
                                               : gemm::HygonBackend::BLAS_OP_N;
    };

    gemm::HygonBackend::blasHandle_t handle{};
    gemm::HygonBackend::blasCreate(&handle);
    gemm::HygonBackend::blasSetStream(
        handle, static_cast<gemm::HygonBackend::stream_t>(this->stream_));

    const auto& alpha_value{alpha.value_or(this->alpha_)};
    const auto& beta_value{beta.value_or(this->beta_)};

    const auto& trans_a_value{trans_a.value_or(this->trans_a_)};
    const auto& trans_b_value{trans_b.value_or(this->trans_b_)};
    auto op_a{get_op_a(trans_a_value, trans_b_value)};
    auto op_b{get_op_b(trans_a_value, trans_b_value)};
    const void* alpha_ptr{this->GetAlphaPtr(alpha_value, c.dtype())};
    const void* beta_ptr{this->GetBetaPtr(beta_value, c.dtype())};

    if (this->batch_count_ == 1) {
      gemm::HygonBackend::blasGemmEx(
          handle, op_a, op_b, swap_a_and_b ? this->n_ : this->m_,
          swap_a_and_b ? this->m_ : this->n_, this->k_, alpha_ptr,
          swap_a_and_b ? b.data() : a.data(),
          gemm::HygonBackend::GetDataType(swap_a_and_b ? b.dtype()
                                                       : a.dtype()),
          swap_a_and_b ? this->ldb_ : this->lda_,
          swap_a_and_b ? a.data() : b.data(),
          gemm::HygonBackend::GetDataType(swap_a_and_b ? a.dtype()
                                                       : b.dtype()),
          swap_a_and_b ? this->lda_ : this->ldb_, beta_ptr, c.data(),
          gemm::HygonBackend::GetDataType(c.dtype()), this->ldc_,
          gemm::HygonBackend::GetComputeType(c.dtype()),
          gemm::HygonBackend::BLAS_GEMM_DEFAULT);
    } else {
      gemm::HygonBackend::blasGemmStridedBatchedEx(
          handle, op_a, op_b, swap_a_and_b ? this->n_ : this->m_,
          swap_a_and_b ? this->m_ : this->n_, this->k_, alpha_ptr,
          swap_a_and_b ? b.data() : a.data(),
          gemm::HygonBackend::GetDataType(swap_a_and_b ? b.dtype()
                                                       : a.dtype()),
          swap_a_and_b ? this->ldb_ : this->lda_,
          swap_a_and_b ? this->batch_stride_b_ : this->batch_stride_a_,
          swap_a_and_b ? a.data() : b.data(),
          gemm::HygonBackend::GetDataType(swap_a_and_b ? a.dtype()
                                                       : b.dtype()),
          swap_a_and_b ? this->lda_ : this->ldb_,
          swap_a_and_b ? this->batch_stride_a_ : this->batch_stride_b_,
          beta_ptr, c.data(), gemm::HygonBackend::GetDataType(c.dtype()),
          this->ldc_, this->batch_stride_c_, this->batch_count_,
          gemm::HygonBackend::GetComputeType(c.dtype()),
          gemm::HygonBackend::BLAS_GEMM_DEFAULT);
    }

    gemm::HygonBackend::blasDestroy(handle);
  }
};

}  // namespace infini::ops

#endif
