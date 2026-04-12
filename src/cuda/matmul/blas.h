#ifndef INFINI_OPS_CUDA_MATMUL_BLAS_H_
#define INFINI_OPS_CUDA_MATMUL_BLAS_H_

#include <utility>

#include "base/matmul.h"
#include "cuda/blas_utils.h"

namespace infini::ops {

template <typename Backend>
class BlasMatmul : public Matmul {
 public:
  BlasMatmul(const Tensor a, const Tensor b, Tensor c, bool trans_a,
             bool trans_b)
      : Matmul{a, b, c, trans_a, trans_b},
        a_is_col_major_{a.stride(-1) == 1},
        b_is_col_major_{b.stride(-1) == 1},
        swap_a_and_b_{c.stride(-1) == 1} {
    // TODO: Check constraints.
  }

  BlasMatmul(const Tensor a, const Tensor b, Tensor c)
      : BlasMatmul{a, b, c, false, false} {}

  void operator()(const Tensor a, const Tensor b, Tensor c, bool trans_a,
                  bool trans_b) const override {
    Backend::BlasSetStream(GetHandle(),
                           static_cast<typename Backend::Stream>(stream_));

    auto op_a{GetOpA(trans_a, trans_b)};
    auto op_b{GetOpB(trans_a, trans_b)};

    const float alpha{1.0f};
    const float beta{0.0f};

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
        swap_a_and_b_ ? batch_stride_a_ : batch_stride_b_, &beta, c.data(),
        BlasUtils<Backend::kDeviceType>::GetDataType(c.dtype()), ldc_,
        batch_stride_c_, batch_count_,
        BlasUtils<Backend::kDeviceType>::GetComputeType(c.dtype()),
        Backend::BLAS_GEMM_DEFAULT);
  }

 private:
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

  // TODO: This static singleton is not thread-safe under concurrent access
  // from multiple host threads. Add proper synchronization in the future.
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
