#ifndef INFINI_OPS_CAMBRICON_GEMM_CNBLAS_H_
#define INFINI_OPS_CAMBRICON_GEMM_CNBLAS_H_

#include <cassert>
#include <memory>
#include <vector>

// clang-format off
#include <cnnl.h>
#include <cnrt.h>
// clang-format on

#include "base/gemm.h"
#include "cambricon/common.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kCambricon> : public Gemm {
 public:
  Operator(const Tensor a, const Tensor b, std::optional<float> alpha,
           std::optional<float> beta, std::optional<int> trans_a,
           std::optional<int> trans_b, Tensor c)
      : Gemm{a, b, alpha, beta, trans_a, trans_b, c},
        a_rows_{a.size(-2)},
        a_cols_{a.size(-1)},
        b_rows_{b.size(-2)},
        b_cols_{b.size(-1)},
        c_rows_{c.size(-2)},
        c_cols_{c.size(-1)} {
    // Currently only support non-transposed matrices
    assert(!trans_a_ && "trans_a=true is not currently supported");
    assert(!trans_b_ && "trans_b=true is not currently supported");
    // Create CNNL handle
    cnnlCreate(&cnnl_handle_);

    // Create tensor descriptors
    cnnlCreateTensorDescriptor(&desc_a_);
    cnnlCreateTensorDescriptor(&desc_b_);
    cnnlCreateTensorDescriptor(&desc_c_);

    // Create matmul descriptor and algo
    cnnlCreateMatMulDescriptor(&matmul_desc_);
    cnnlCreateMatMulAlgo(&matmul_algo_);
    cnnlCreateMatMulHeuristicResult(&heuristic_result_);

    // Set stride usage
    int32_t use_stride = 1;
    cnnlSetMatMulDescAttr(matmul_desc_, CNNL_MATMUL_USE_STRIDE, &use_stride,
                          sizeof(int32_t));

    // Setup tensor descriptors using physical dimensions
    SetupTensorDescriptor(desc_a_, a_strides_, a_type_, a_rows_, a_cols_,
                          batch_count_, batch_stride_a_);
    SetupTensorDescriptor(desc_b_, b_strides_, b_type_, b_rows_, b_cols_,
                          batch_count_, batch_stride_b_);
    SetupTensorDescriptor(desc_c_, c_strides_, c_type_, c_rows_, c_cols_,
                          batch_count_, batch_stride_c_);
    int count = 0;
    cnnlGetBatchMatMulExAlgoHeuristic(cnnl_handle_, matmul_desc_, desc_a_,
                                      desc_b_, desc_c_, NULL, 1,
                                      &heuristic_result_, &count);
    cnnlGetBatchMatMulExHeuristicResult(heuristic_result_, matmul_algo_,
                                        &workspace_size_);
  }

  Operator(const Tensor a, const Tensor b, Tensor c)
      : Operator{a, b, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                 c} {}

  Operator(const Tensor a, const Tensor b, std::optional<float> alpha,
           std::optional<float> beta, Tensor c)
      : Operator{a, b, alpha, beta, std::nullopt, std::nullopt, c} {}

  ~Operator() {
    cnnlDestroyTensorDescriptor(desc_a_);
    cnnlDestroyTensorDescriptor(desc_b_);
    cnnlDestroyTensorDescriptor(desc_c_);
    cnnlDestroyMatMulDescriptor(matmul_desc_);
    cnnlDestroyMatMulAlgo(matmul_algo_);
    cnnlDestroyMatMulHeuristicResult(heuristic_result_);
    cnnlDestroy(cnnl_handle_);
  }

  void operator()(const Tensor a, const Tensor b, std::optional<float> alpha,
                  std::optional<float> beta, std::optional<int> trans_a,
                  std::optional<int> trans_b, Tensor c) const override {
    const auto& alpha_value{alpha.value_or(alpha_)};
    const auto& beta_value{beta.value_or(beta_)};

    // Set queue for this execution
    cnnlSetQueue(cnnl_handle_, (cnrtQueue_t)stream_);

    // Allocate workspace using pre-computed size
    void* workspace = nullptr;
    if (workspace_size_ > 0) {
      cnrtMalloc(&workspace, workspace_size_);
    }

    // Execute batch matrix multiply
    cnnlBatchMatMulEx(cnnl_handle_, matmul_desc_, matmul_algo_, &alpha_value,
                      desc_a_, a.data(), desc_b_, b.data(), &beta_value,
                      desc_c_, c.data(), workspace, workspace_size_);

    // Cleanup workspace
    if (workspace) {
      cnrtFree(workspace);
    }
    cnrtQueueSync((cnrtQueue_t)stream_);
  }

 private:
  void SetupTensorDescriptor(cnnlTensorDescriptor_t desc,
                             const Tensor::Strides& strides, DataType dtype,
                             Tensor::Size rows, Tensor::Size cols,
                             Tensor::Size batch, Tensor::Stride batch_stride) {
    cnnlDataType_t cnnl_dtype = cnnl_utils::GetDtype(dtype);

    if (batch > 1) {
      // Batched tensor: [batch, rows, cols]
      std::vector<int> dims = {static_cast<int>(batch), static_cast<int>(rows),
                               static_cast<int>(cols)};
      std::vector<int> strides_arr = {
          static_cast<int>(batch_stride),
          static_cast<int>(strides[strides.size() - 2]),
          static_cast<int>(strides[strides.size() - 1])};
      cnnlSetTensorDescriptorEx(desc, CNNL_LAYOUT_ARRAY, cnnl_dtype,
                                dims.size(), dims.data(), strides_arr.data());
    } else {
      // 2D tensor: [rows, cols]
      std::vector<int> dims = {static_cast<int>(rows), static_cast<int>(cols)};
      std::vector<int> strides_arr = {
          static_cast<int>(strides[strides.size() - 2]),
          static_cast<int>(strides[strides.size() - 1])};
      cnnlSetTensorDescriptorEx(desc, CNNL_LAYOUT_ARRAY, cnnl_dtype,
                                dims.size(), dims.data(), strides_arr.data());
    }
  }

  cnnlHandle_t cnnl_handle_;
  cnnlTensorDescriptor_t desc_a_;
  cnnlTensorDescriptor_t desc_b_;
  cnnlTensorDescriptor_t desc_c_;
  cnnlMatMulDescriptor_t matmul_desc_;
  cnnlMatMulAlgo_t matmul_algo_;
  cnnlMatMulHeuristicResult_t heuristic_result_;

  // Physical storage dimensions for each tensor
  Tensor::Size a_rows_, a_cols_;
  Tensor::Size b_rows_, b_cols_;
  Tensor::Size c_rows_, c_cols_;
};

}  // namespace infini::ops

#endif
