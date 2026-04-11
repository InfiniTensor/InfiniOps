#ifndef INFINI_OPS_CUDA_LINEAR_KERNEL_H_
#define INFINI_OPS_CUDA_LINEAR_KERNEL_H_

#include <cstddef>

#include "base/linear.h"
#include "cuda/blas_utils.h"
#include "cuda/linear/kernel.cuh"
#include "cuda/runtime_utils.h"

namespace infini::ops {

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
    Backend::BlasSetStream(GetHandle(),
                           static_cast<typename Backend::Stream>(stream_));

    float alpha = 1.0f;
    float beta = 0.0f;

    auto op_a = GetOpA(trans_a, trans_b);
    auto op_b = GetOpB(trans_a, trans_b);

    Backend::BlasGemmStridedBatchedEx(
        GetHandle(), op_a, op_b, swap_a_and_b_ ? n_ : m_,
        swap_a_and_b_ ? m_ : n_, k_, &alpha,
        swap_a_and_b_ ? b.data() : a.data(),
        BlasUtils<Backend::kDeviceType>::GetDataType(swap_a_and_b_ ? b.dtype()
                                                                   : a.dtype()),
        swap_a_and_b_ ? ldb_ : lda_,
        swap_a_and_b_ ? batch_stride_b_ : batch_stride_a_,
        swap_a_and_b_ ? a.data() : b.data(),
        BlasUtils<Backend::kDeviceType>::GetDataType(swap_a_and_b_ ? a.dtype()
                                                                   : b.dtype()),
        swap_a_and_b_ ? lda_ : ldb_,
        swap_a_and_b_ ? batch_stride_a_ : batch_stride_b_, &beta, out.data(),
        BlasUtils<Backend::kDeviceType>::GetDataType(out.dtype()), ldc_,
        batch_stride_c_, batch_count_,
        BlasUtils<Backend::kDeviceType>::GetComputeType(out.dtype()),
        Backend::BLAS_GEMM_DEFAULT);

    if (has_bias_ && bias.has_value()) {
      LaunchBiasAdd(out, bias.value());
    }
  }

 private:
  void LaunchBiasAdd(Tensor out, const Tensor bias) const {
    size_t rows = batch_count_ * m_;
    size_t cols = n_;
    size_t total = rows * cols;

    int block_size = RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();

    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    DispatchFunc<Backend::kDeviceType, AllFloatTypes>(
        out.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          dim3 blockDims(block_size);
          dim3 gridDims((total + block_size - 1) / block_size);

          BiasAddKernel<T><<<gridDims, blockDims, 0, cuda_stream>>>(
              reinterpret_cast<T*>(out.data()),
              reinterpret_cast<const T*>(bias.data()), rows, cols);
        },
        "CudaLinear::BiasAdd");
  }

  auto GetOpA(bool trans_a, bool trans_b) const {
    if (swap_a_and_b_) {
      return (b_is_col_major_ == trans_b) ? Backend::BLAS_OP_T
                                          : Backend::BLAS_OP_N;
    }

    return (a_is_col_major_ != trans_a) ? Backend::BLAS_OP_T
                                        : Backend::BLAS_OP_N;
  }

  auto GetOpB(bool trans_a, bool trans_b) const {
    if (swap_a_and_b_) {
      return (a_is_col_major_ == trans_a) ? Backend::BLAS_OP_T
                                          : Backend::BLAS_OP_N;
    }

    return (b_is_col_major_ != trans_b) ? Backend::BLAS_OP_T
                                        : Backend::BLAS_OP_N;
  }

  static typename Backend::BlasHandle& GetHandle() {
    static typename Backend::BlasHandle handle = []() {
      typename Backend::BlasHandle h;
      Backend::BlasCreate(&h);
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
