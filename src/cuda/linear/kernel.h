#ifndef INFINI_OPS_CUDA_LINEAR_KERNEL_H_
#define INFINI_OPS_CUDA_LINEAR_KERNEL_H_

#include <cassert>
#include <cstddef>
#include <cstdint>

// clang-format off
#include "cublasLt.h"
// clang-format on

#include "base/linear.h"
#include "cuda/linear/kernel.cuh"
#include "cuda/runtime_utils.h"
#include "nvidia/blas_utils.h"

namespace infini::ops {

// Linear operator using cuBLASLt with heuristic algorithm selection.
// Computes out = a @ b (+ bias), with optional transpose.
template <typename Backend>
class CudaLinear : public Linear {
 public:
  CudaLinear(const Tensor a, const Tensor b, std::optional<Tensor> bias,
             bool trans_a, bool trans_b, Tensor out)
      : Linear{a, b, bias, trans_a, trans_b, out},
        a_is_col_major_{a.stride(-1) == 1},
        b_is_col_major_{b.stride(-1) == 1},
        swap_a_and_b_{out.stride(-1) == 1} {}

  void operator()(const Tensor a, const Tensor b, std::optional<Tensor> bias,
                  bool trans_a, bool trans_b, Tensor out) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    float alpha = 1.0f;
    float beta = 0.0f;

    auto op_a = GetOpA(trans_a, trans_b);
    auto op_b = GetOpB(trans_a, trans_b);

    auto matmul_m = static_cast<int64_t>(swap_a_and_b_ ? n_ : m_);
    auto matmul_n = static_cast<int64_t>(swap_a_and_b_ ? m_ : n_);
    auto matmul_k = static_cast<int64_t>(k_);

    const auto* a_ptr = swap_a_and_b_ ? b.data() : a.data();
    const auto* b_ptr = swap_a_and_b_ ? a.data() : b.data();
    auto a_dtype =
        BlasUtils<Backend::kDeviceType>::GetDataType(
            swap_a_and_b_ ? b.dtype() : a.dtype());
    auto b_dtype =
        BlasUtils<Backend::kDeviceType>::GetDataType(
            swap_a_and_b_ ? a.dtype() : b.dtype());
    auto c_dtype =
        BlasUtils<Backend::kDeviceType>::GetDataType(out.dtype());
    auto a_ld = static_cast<uint64_t>(swap_a_and_b_ ? ldb_ : lda_);
    auto b_ld = static_cast<uint64_t>(swap_a_and_b_ ? lda_ : ldb_);
    auto c_ld = static_cast<uint64_t>(ldc_);
    auto a_batch_stride = static_cast<int64_t>(
        swap_a_and_b_ ? batch_stride_b_ : batch_stride_a_);
    auto b_batch_stride = static_cast<int64_t>(
        swap_a_and_b_ ? batch_stride_a_ : batch_stride_b_);
    auto c_batch_stride = static_cast<int64_t>(batch_stride_c_);

    // Create cuBLASLt matmul descriptor.
    cublasLtMatmulDesc_t op_desc{};
    auto status = cublasLtMatmulDescCreate(
        &op_desc,
        BlasUtils<Backend::kDeviceType>::GetComputeType(out.dtype()),
        CUDA_R_32F);
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to create cuBLASLt matmul descriptor");

    status = cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
    assert(status == CUBLAS_STATUS_SUCCESS);

    status = cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
    assert(status == CUBLAS_STATUS_SUCCESS);

    // Create matrix layouts.
    cublasLtMatrixLayout_t a_layout{};
    status = cublasLtMatrixLayoutCreate(
        &a_layout, a_dtype,
        op_a == CUBLAS_OP_N ? matmul_m : matmul_k,
        op_a == CUBLAS_OP_N ? matmul_k : matmul_m, a_ld);
    assert(status == CUBLAS_STATUS_SUCCESS);

    cublasLtMatrixLayout_t b_layout{};
    status = cublasLtMatrixLayoutCreate(
        &b_layout, b_dtype,
        op_b == CUBLAS_OP_N ? matmul_k : matmul_n,
        op_b == CUBLAS_OP_N ? matmul_n : matmul_k, b_ld);
    assert(status == CUBLAS_STATUS_SUCCESS);

    cublasLtMatrixLayout_t c_layout{};
    status = cublasLtMatrixLayoutCreate(
        &c_layout, c_dtype, matmul_m, matmul_n, c_ld);
    assert(status == CUBLAS_STATUS_SUCCESS);

    if (batch_count_ > 1) {
      SetStridedBatchAttributes(a_layout, a_batch_stride);
      SetStridedBatchAttributes(b_layout, b_batch_stride);
      SetStridedBatchAttributes(c_layout, c_batch_stride);
    }

    // Search for optimal algorithm.
    cublasLtMatmulPreference_t preference{};
    status = cublasLtMatmulPreferenceCreate(&preference);
    assert(status == CUBLAS_STATUS_SUCCESS);

    size_t workspace_size = workspace_size_in_bytes_;
    status = cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size));
    assert(status == CUBLAS_STATUS_SUCCESS);

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned_results = 0;
    status = cublasLtMatmulAlgoGetHeuristic(
        GetHandle(), op_desc, a_layout, b_layout, c_layout, c_layout,
        preference, 1, &heuristic, &returned_results);
    assert(status == CUBLAS_STATUS_SUCCESS && returned_results > 0 &&
           "failed to find a cuBLASLt algorithm for Linear");

    // Execute.
    status = cublasLtMatmul(
        GetHandle(), op_desc, &alpha, a_ptr, a_layout, b_ptr, b_layout,
        &beta, out.data(), c_layout, out.data(), c_layout,
        &heuristic.algo, workspace_, workspace_size_in_bytes_, cuda_stream);
    assert(status == CUBLAS_STATUS_SUCCESS && "cuBLASLt Linear matmul failed");

    // Cleanup.
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(c_layout);
    cublasLtMatrixLayoutDestroy(b_layout);
    cublasLtMatrixLayoutDestroy(a_layout);
    cublasLtMatmulDescDestroy(op_desc);

    // Bias add.
    if (has_bias_ && bias.has_value()) {
      LaunchBiasAdd(out, bias.value(), cuda_stream);
    }
  }

 private:
  void LaunchBiasAdd(Tensor out, const Tensor bias,
                     typename Backend::Stream stream) const {
    size_t rows = batch_count_ * m_;
    size_t cols = n_;
    size_t total = rows * cols;
    int block_size = RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();

    DispatchFunc<Backend::kDeviceType, AllFloatTypes>(
        out.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          dim3 blockDims(block_size);
          dim3 gridDims((total + block_size - 1) / block_size);

          BiasAddKernel<T><<<gridDims, blockDims, 0, stream>>>(
              reinterpret_cast<T*>(out.data()),
              reinterpret_cast<const T*>(bias.data()), rows, cols);
        },
        "CudaLinear::BiasAdd");
  }

  void SetStridedBatchAttributes(cublasLtMatrixLayout_t layout,
                                 int64_t batch_stride) const {
    int batch_count = static_cast<int>(batch_count_);
    auto status = cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count, sizeof(batch_count));
    assert(status == CUBLAS_STATUS_SUCCESS);

    status = cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batch_stride, sizeof(batch_stride));
    assert(status == CUBLAS_STATUS_SUCCESS);
  }

  cublasOperation_t GetOpA(bool trans_a, bool trans_b) const {
    if (swap_a_and_b_) {
      return (b_is_col_major_ == trans_b) ? CUBLAS_OP_T : CUBLAS_OP_N;
    }

    return (a_is_col_major_ != trans_a) ? CUBLAS_OP_T : CUBLAS_OP_N;
  }

  cublasOperation_t GetOpB(bool trans_a, bool trans_b) const {
    if (swap_a_and_b_) {
      return (a_is_col_major_ == trans_a) ? CUBLAS_OP_T : CUBLAS_OP_N;
    }

    return (b_is_col_major_ != trans_b) ? CUBLAS_OP_T : CUBLAS_OP_N;
  }

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

  bool a_is_col_major_{false};

  bool b_is_col_major_{false};

  bool swap_a_and_b_{false};
};

}  // namespace infini::ops

#endif
