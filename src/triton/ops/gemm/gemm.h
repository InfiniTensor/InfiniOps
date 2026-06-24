#ifndef INFINI_OPS_TRITON_GEMM_H_
#define INFINI_OPS_TRITON_GEMM_H_

#include <cuda.h>

#include <cassert>
#include <cstdint>
#include <optional>

#include "base/gemm.h"
#include "data_type.h"
#include "gemm/infini_ops_triton_gemm.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kNvidia, 8> : public Gemm {
 public:
  using Gemm::operator();

  Operator(const Tensor a, const Tensor b, std::optional<float> alpha,
           std::optional<float> beta, std::optional<int> trans_a,
           std::optional<int> trans_b, Tensor c)
      : Gemm{a, b, alpha, beta, trans_a, trans_b, c} {}

  void operator()(const Tensor a, const Tensor b, std::optional<float> alpha,
                  std::optional<float> beta, std::optional<int> trans_a,
                  std::optional<int> trans_b, Tensor c) const override {
    assert(a_type_ == b_type_ && b_type_ == c_type_ &&
           "Triton `Gemm` requires A, B, and C tensors to have the same dtype");

    const auto alpha_value = alpha.value_or(alpha_);
    const auto beta_value = beta.value_or(beta_);
    const auto trans_a_value = static_cast<bool>(trans_a.value_or(trans_a_));
    const auto trans_b_value = static_cast<bool>(trans_b.value_or(trans_b_));

    const auto stride_am =
        static_cast<int64_t>(trans_a_value ? a_strides_[a_strides_.size() - 1]
                                           : a_strides_[a_strides_.size() - 2]);
    const auto stride_ak =
        static_cast<int64_t>(trans_a_value ? a_strides_[a_strides_.size() - 2]
                                           : a_strides_[a_strides_.size() - 1]);
    const auto stride_bk =
        static_cast<int64_t>(trans_b_value ? b_strides_[b_strides_.size() - 1]
                                           : b_strides_[b_strides_.size() - 2]);
    const auto stride_bn =
        static_cast<int64_t>(trans_b_value ? b_strides_[b_strides_.size() - 2]
                                           : b_strides_[b_strides_.size() - 1]);
    const auto stride_cm =
        static_cast<int64_t>(c_strides_[c_strides_.size() - 2]);
    const auto stride_cn =
        static_cast<int64_t>(c_strides_[c_strides_.size() - 1]);
    const auto batch_stride_a = static_cast<int64_t>(batch_stride_a_);
    const auto batch_stride_b = static_cast<int64_t>(batch_stride_b_);
    const auto batch_stride_c = static_cast<int64_t>(batch_stride_c_);
    const auto batch_count = static_cast<int32_t>(batch_count_);

    load_infini_ops_triton_gemm(c_type_);

    auto result = launch_infini_ops_triton_gemm(
        c_type_, static_cast<CUstream>(stream_),
        reinterpret_cast<CUdeviceptr>(const_cast<void*>(a.data())),
        reinterpret_cast<CUdeviceptr>(const_cast<void*>(b.data())),
        reinterpret_cast<CUdeviceptr>(c.data()), alpha_value, beta_value,
        static_cast<int32_t>(m_), static_cast<int32_t>(n_),
        static_cast<int32_t>(k_), stride_am, stride_ak, stride_bk, stride_bn,
        stride_cm, stride_cn, batch_stride_a, batch_stride_b, batch_stride_c,
        batch_count);

    assert(result == CUDA_SUCCESS && "Triton `Gemm` launch failed");
  }
};

}  // namespace infini::ops

#endif
