#ifndef INFINI_OPS_NVIDIA_GEMM_CUBLASLT_H_
#define INFINI_OPS_NVIDIA_GEMM_CUBLASLT_H_

#include <cassert>
#include <cstdint>

// clang-format off
#include "cublasLt.h"
// clang-format on

#include "base/gemm.h"
#include "cuda/nvidia/blas_utils.h"
#include "cuda/nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kNvidia, 1> : public Gemm {
 public:
  Operator(const Tensor a, const Tensor b, std::optional<float> alpha,
           std::optional<float> beta, std::optional<int> trans_a,
           std::optional<int> trans_b, Tensor c)
      : Gemm{a, b, alpha, beta, trans_a, trans_b, c},
        a_is_col_major_{a.stride(-1) == 1},
        b_is_col_major_{b.stride(-1) == 1},
        swap_a_and_b_{c.stride(-1) == 1} {
    // Everything below is a function of stored shape / strides / dtype /
    // transpose flags — constant for the lifetime of this cached Operator
    // instance. Building the `cublasLt` descriptors and picking the
    // heuristic here keeps `operator()` down to a single `cublasLtMatmul`
    // launch. A hot-path decode step saves ~10µs/call × 64 linears/step.
    const auto op_a_init = GetOpA(trans_a_, trans_b_);
    const auto op_b_init = GetOpB(trans_a_, trans_b_);
    const auto matmul_m = static_cast<int64_t>(swap_a_and_b_ ? n_ : m_);
    const auto matmul_n = static_cast<int64_t>(swap_a_and_b_ ? m_ : n_);
    const auto matmul_k = static_cast<int64_t>(k_);

    const auto a_dtype_init = BlasUtils<Device::Type::kNvidia>::GetDataType(
        swap_a_and_b_ ? b.dtype() : a.dtype());
    const auto b_dtype_init = BlasUtils<Device::Type::kNvidia>::GetDataType(
        swap_a_and_b_ ? a.dtype() : b.dtype());
    const auto c_dtype_init =
        BlasUtils<Device::Type::kNvidia>::GetDataType(c.dtype());
    const auto a_ld_init =
        static_cast<uint64_t>(swap_a_and_b_ ? ldb_ : lda_);
    const auto b_ld_init =
        static_cast<uint64_t>(swap_a_and_b_ ? lda_ : ldb_);
    const auto c_ld_init = static_cast<uint64_t>(ldc_);

    auto status = cublasLtMatmulDescCreate(
        &op_desc_,
        BlasUtils<Device::Type::kNvidia>::GetComputeType(c.dtype()),
        CUDA_R_32F);
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to create cuBLASLt matmul descriptor");

    status = cublasLtMatmulDescSetAttribute(
        op_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &op_a_init, sizeof(op_a_init));
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to set cuBLASLt transa attribute");

    status = cublasLtMatmulDescSetAttribute(
        op_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &op_b_init, sizeof(op_b_init));
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to set cuBLASLt transb attribute");

    status = cublasLtMatrixLayoutCreate(
        &a_layout_, a_dtype_init,
        op_a_init == CUBLAS_OP_N ? matmul_m : matmul_k,
        op_a_init == CUBLAS_OP_N ? matmul_k : matmul_m, a_ld_init);
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to create cuBLASLt A layout");

    status = cublasLtMatrixLayoutCreate(
        &b_layout_, b_dtype_init,
        op_b_init == CUBLAS_OP_N ? matmul_k : matmul_n,
        op_b_init == CUBLAS_OP_N ? matmul_n : matmul_k, b_ld_init);
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to create cuBLASLt B layout");

    status = cublasLtMatrixLayoutCreate(&c_layout_, c_dtype_init, matmul_m,
                                        matmul_n, c_ld_init);
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to create cuBLASLt C layout");

    if (batch_count_ > 1) {
      const auto a_batch_stride = static_cast<int64_t>(
          swap_a_and_b_ ? batch_stride_b_ : batch_stride_a_);
      const auto b_batch_stride = static_cast<int64_t>(
          swap_a_and_b_ ? batch_stride_a_ : batch_stride_b_);
      const auto c_batch_stride = static_cast<int64_t>(batch_stride_c_);
      SetStridedBatchAttributes(a_layout_, a_batch_stride);
      SetStridedBatchAttributes(b_layout_, b_batch_stride);
      SetStridedBatchAttributes(c_layout_, c_batch_stride);
    }

    status = cublasLtMatmulPreferenceCreate(&preference_);
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to create cuBLASLt preference");

    // Handle's default workspace is `nullptr` / 0 bytes; pick an algo
    // consistent with that. If a future caller supplies a real workspace,
    // this will still work — heuristic-picked algos with `workspace_size = 0`
    // run correctly even when a larger workspace is available.
    const std::uint64_t pref_workspace_size = 0;
    status = cublasLtMatmulPreferenceSetAttribute(
        preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &pref_workspace_size, sizeof(pref_workspace_size));
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to set cuBLASLt workspace preference");

    int returned_results{0};
    status = cublasLtMatmulAlgoGetHeuristic(
        GetHandle(), op_desc_, a_layout_, b_layout_, c_layout_, c_layout_,
        preference_, 1, &heuristic_, &returned_results);
    assert(status == CUBLAS_STATUS_SUCCESS && returned_results > 0 &&
           "failed to find a cuBLASLt GEMM algorithm");
  }

  Operator(const Tensor a, const Tensor b, std::optional<float> alpha,
           std::optional<float> beta, Tensor c)
      : Operator{a, b, alpha, beta, std::nullopt, std::nullopt, c} {}

  Operator(const Tensor a, const Tensor b, Tensor c)
      : Operator{a, b, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                 c} {}

  ~Operator() {
    if (preference_) cublasLtMatmulPreferenceDestroy(preference_);
    if (c_layout_) cublasLtMatrixLayoutDestroy(c_layout_);
    if (b_layout_) cublasLtMatrixLayoutDestroy(b_layout_);
    if (a_layout_) cublasLtMatrixLayoutDestroy(a_layout_);
    if (op_desc_) cublasLtMatmulDescDestroy(op_desc_);
  }

  Operator(const Operator&) = delete;
  Operator& operator=(const Operator&) = delete;

  void operator()(const Tensor a, const Tensor b, std::optional<float> alpha,
                  std::optional<float> beta, std::optional<int> /*trans_a*/,
                  std::optional<int> /*trans_b*/, Tensor c) const override {
    // `trans_a`/`trans_b` are part of the `Operator::Call` cache key, so any
    // override already routed to this instance matches the descriptor we
    // built in the constructor; ignore the per-call values here.
    const auto alpha_value{alpha.value_or(alpha_)};
    const auto beta_value{beta.value_or(beta_)};

    const auto* a_ptr{swap_a_and_b_ ? b.data() : a.data()};
    const auto* b_ptr{swap_a_and_b_ ? a.data() : b.data()};

    auto status = cublasLtMatmul(
        GetHandle(), op_desc_, GetAlphaPtr(alpha_value), a_ptr, a_layout_,
        b_ptr, b_layout_, GetBetaPtr(beta_value), c.data(), c_layout_,
        c.data(), c_layout_, &heuristic_.algo, workspace_,
        workspace_size_in_bytes_,
        static_cast<Runtime<Device::Type::kNvidia>::Stream>(stream_));
    assert(status == CUBLAS_STATUS_SUCCESS && "cuBLASLt GEMM launch failed");
  }

 private:
  static cublasLtHandle_t& GetHandle() {
    static cublasLtHandle_t handle = []() {
      cublasLtHandle_t h{};
      auto status = cublasLtCreate(&h);
      assert(status == CUBLAS_STATUS_SUCCESS &&
             "failed to create cuBLASLt handle");
      return h;
    }();
    return handle;
  }

  const void* GetAlphaPtr(const float& alpha) const { return &alpha; }

  const void* GetBetaPtr(const float& beta) const { return &beta; }

  void SetStridedBatchAttributes(cublasLtMatrixLayout_t layout,
                                 int64_t batch_stride) const {
    const int batch_count{static_cast<int>(batch_count_)};
    auto status = cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count));
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to set cuBLASLt batch count");

    status = cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride,
        sizeof(batch_stride));
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to set cuBLASLt batch stride");
  }

  cublasOperation_t GetOpA(int trans_a, int trans_b) const {
    if (swap_a_and_b_) {
      return (b_is_col_major_ == trans_b) ? CUBLAS_OP_T : CUBLAS_OP_N;
    }
    return (a_is_col_major_ != trans_a) ? CUBLAS_OP_T : CUBLAS_OP_N;
  }

  cublasOperation_t GetOpB(int trans_a, int trans_b) const {
    if (swap_a_and_b_) {
      return (a_is_col_major_ == trans_a) ? CUBLAS_OP_T : CUBLAS_OP_N;
    }
    return (b_is_col_major_ != trans_b) ? CUBLAS_OP_T : CUBLAS_OP_N;
  }

  bool a_is_col_major_{false};

  bool b_is_col_major_{false};

  bool swap_a_and_b_{false};

  // cuBLASLt state created once per cached `Operator` instance.
  cublasLtMatmulDesc_t op_desc_{};
  cublasLtMatrixLayout_t a_layout_{};
  cublasLtMatrixLayout_t b_layout_{};
  cublasLtMatrixLayout_t c_layout_{};
  cublasLtMatmulPreference_t preference_{};
  cublasLtMatmulHeuristicResult_t heuristic_{};
};

}  // namespace infini::ops

#endif
