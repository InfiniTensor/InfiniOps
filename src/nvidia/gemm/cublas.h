#ifndef INFINI_OPS_NVIDIA_GEMM_CUBLAS_H_
#define INFINI_OPS_NVIDIA_GEMM_CUBLAS_H_

#include <utility>

// clang-format off
#include "cublas_v2.h"
// clang-format on

#include "base/gemm.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::kNvidia> : public Gemm {
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

  void operator()(void* stream, const void* a, const void* b, void* c) const {
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetStream(handle, static_cast<cudaStream_t>(stream));

    // TODO: Add support for more data types.
    assert(a_type_ == kFloat32 && b_type_ == kFloat32 && c_type_ == kFloat32 &&
           "`operator()` not implemented for this data type");

    cublasGemmEx(handle, trans_a_ ? CUBLAS_OP_T : CUBLAS_OP_N,
                 trans_b_ ? CUBLAS_OP_T : CUBLAS_OP_N, m_, n_, k_, &alpha_, b,
                 CUDA_R_32F, lda_, a, CUDA_R_32F, ldb_, &beta_, c, CUDA_R_32F,
                 ldc_, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);

    cublasDestroy(handle);
  }

 private:
  Tensor::Stride lda_{0};

  Tensor::Stride ldb_{0};

  Tensor::Stride ldc_{0};
};

}  // namespace infini::ops

#endif
