#ifndef INFINI_OPS_CAMBRICON_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_
#define INFINI_OPS_CAMBRICON_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_

#include <cnnl_extra.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "base/flash_attn_varlen_func.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops {
namespace cambricon_flash_attn_detail {

struct FlashAttentionDescriptorDeleter {
  using pointer = cnnlFlashAttentionDescriptor_t;

  void operator()(pointer desc) const noexcept {
    if (desc) {
      (void)cnnlDestroyFlashAttentionDescriptor(desc);
    }
  }
};

using FlashAttentionDescriptor =
    std::unique_ptr<std::remove_pointer_t<cnnlFlashAttentionDescriptor_t>,
                    FlashAttentionDescriptorDeleter>;

inline FlashAttentionDescriptor CreateFlashAttentionDescriptor() {
  cnnlFlashAttentionDescriptor_t desc{nullptr};
  INFINI_OPS_CNNL_CHECK(cnnlCreateFlashAttentionDescriptor(&desc));
  return FlashAttentionDescriptor{desc};
}

}  // namespace cambricon_flash_attn_detail

template <>
class Operator<FlashAttnVarlenFunc, Device::Type::kCambricon>
    : public FlashAttnVarlenFunc {
 public:
  static bool supports(
      const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
      const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,
      const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
      const int64_t max_seqlen_k, const double dropout_p,
      const std::optional<double> softmax_scale, const bool causal,
      const std::vector<int64_t> window_size, const double softcap,
      const bool deterministic, const bool return_attn_probs, Tensor out,
      std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask) {
    if (!FlashAttnVarlenFunc::supports(
            q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes, block_table,
            max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
            window_size, softcap, deterministic, return_attn_probs, out,
            softmax_lse, s_dmask)) {
      return false;
    }
    if (alibi_slopes || block_table || softmax_lse || s_dmask ||
        return_attn_probs || deterministic || dropout_p != 0.0 ||
        softcap != 0.0 || causal != true ||
        window_size != std::vector<int64_t>{-1, -1}) {
      return false;
    }
    if (q.dtype() != DataType::kFloat16 && q.dtype() != DataType::kBFloat16 &&
        q.dtype() != DataType::kFloat32) {
      return false;
    }
    if (q.dtype() == DataType::kFloat32 &&
        (q.shape()[2] > 256 || v.shape()[2] > 256)) {
      return false;
    }
    if (q.shape()[2] > 512 || v.shape()[2] > 512 ||
        q.shape()[1] % k.shape()[1] != 0) {
      return false;
    }
    if (!q.IsContiguous() || !k.IsContiguous() || !v.IsContiguous() ||
        !out.IsContiguous() || !cu_seqlens_q.IsContiguous() ||
        !cu_seqlens_k.IsContiguous()) {
      return false;
    }
    if (cu_seqlens_q.dtype() != DataType::kInt32 ||
        cu_seqlens_k.dtype() != DataType::kInt32) {
      return false;
    }
    if (max_seqlen_q <= 0 || max_seqlen_k <= 0 ||
        max_seqlen_q > std::numeric_limits<int>::max() ||
        max_seqlen_k > std::numeric_limits<int>::max()) {
      return false;
    }
    const auto scale = softmax_scale.value_or(
        1.0 / std::sqrt(static_cast<double>(q.shape()[2])));
    return std::isfinite(scale) && scale > 0.0 && scale <= 1.0;
  }

  Operator(const Tensor q, const Tensor k, const Tensor v,
           const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
           const std::optional<Tensor> alibi_slopes,
           const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
           const int64_t max_seqlen_k, const double dropout_p,
           const std::optional<double> softmax_scale, const bool causal,
           const std::vector<int64_t> window_size, const double softcap,
           const bool deterministic, const bool return_attn_probs, Tensor out,
           std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask)
      : FlashAttnVarlenFunc{q,
                            k,
                            v,
                            cu_seqlens_q,
                            cu_seqlens_k,
                            alibi_slopes,
                            block_table,
                            max_seqlen_q,
                            max_seqlen_k,
                            dropout_p,
                            softmax_scale,
                            causal,
                            window_size,
                            softcap,
                            deterministic,
                            return_attn_probs,
                            out,
                            softmax_lse,
                            s_dmask} {
    if (!supports(q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes,
                  block_table, max_seqlen_q, max_seqlen_k, dropout_p,
                  softmax_scale, causal, window_size, softcap, deterministic,
                  return_attn_probs, out, softmax_lse, s_dmask)) {
      throw std::invalid_argument(
          "Cambricon FlashAttnVarlenFunc does not support these parameters");
    }

    scale_ = static_cast<float>(softmax_scale.value_or(
        1.0 / std::sqrt(static_cast<double>(q.shape()[2]))));
    cnnl_handle_ = cnnl_utils::CreateHandle();
    flash_desc_ = cambricon_flash_attn_detail::CreateFlashAttentionDescriptor();
    INFINI_OPS_CNNL_CHECK(cnnlSetFlashAttentionDescriptor_v2(
        flash_desc_.get(), CNNL_DTYPE_FLOAT, CNNL_COMPUTATION_HIGH_PRECISION,
        CNNL_ATTN_MASK_CAUSAL, CNNL_ATTN_TENSOR_LAYOUT_PACKED, true, false,
        false, static_cast<int>(max_seqlen_q), static_cast<int>(max_seqlen_k),
        0.0F, scale_));

    q_desc_ = cnnl_utils::MakeTensorDescriptor(q_dtype_, q_shape_, q_strides_);
    k_desc_ = cnnl_utils::MakeTensorDescriptor(k_dtype_, k_shape_, k_strides_);
    v_desc_ = cnnl_utils::MakeTensorDescriptor(v_dtype_, v_shape_, v_strides_);
    cu_q_desc_ = cnnl_utils::MakeTensorDescriptor(
        cu_seqlens_q_dtype_, cu_seqlens_q_shape_, cu_seqlens_q_strides_);
    cu_k_desc_ = cnnl_utils::MakeTensorDescriptor(
        cu_seqlens_k_dtype_, cu_seqlens_k_shape_, cu_seqlens_k_strides_);
    out_desc_ =
        cnnl_utils::MakeTensorDescriptor(out_dtype_, out_shape_, out_strides_);

    INFINI_OPS_CNNL_CHECK(cnnlGetFlashAttentionForwardWorkspaceSize_v3(
        cnnl_handle_.get(), flash_desc_.get(), q_desc_.get(), k_desc_.get(),
        v_desc_.get(), cu_q_desc_.get(), cu_k_desc_.get(), nullptr, nullptr,
        nullptr, nullptr, out_desc_.get(), &workspace_size_));
    default_workspace_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_);
  }

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<Tensor> block_table,
                  const int64_t max_seqlen_q, const int64_t max_seqlen_k,
                  const double dropout_p,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool deterministic, const bool return_attn_probs,
                  Tensor out, std::optional<Tensor> softmax_lse,
                  std::optional<Tensor> s_dmask) const override {
    if (!supports(q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes,
                  block_table, max_seqlen_q, max_seqlen_k, dropout_p,
                  softmax_scale, causal, window_size, softcap, deterministic,
                  return_attn_probs, out, softmax_lse, s_dmask) ||
        q.shape() != q_shape_ || k.shape() != k_shape_ ||
        v.shape() != v_shape_ || out.shape() != out_shape_ ||
        cu_seqlens_q.shape() != cu_seqlens_q_shape_ ||
        cu_seqlens_k.shape() != cu_seqlens_k_shape_ || q.dtype() != q_dtype_ ||
        k.dtype() != k_dtype_ || v.dtype() != v_dtype_ ||
        out.dtype() != out_dtype_ || q.device().index() != device_index_) {
      throw std::invalid_argument(
          "Cambricon FlashAttnVarlenFunc prepared metadata changed");
    }

    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(
        cnnl_handle_.get(), static_cast<cnrtQueue_t>(stream_ ? stream_ : 0)));
    void* workspace = workspace_ ? workspace_ : default_workspace_.get();
    const auto available_workspace =
        workspace_ ? workspace_size_in_bytes_ : workspace_size_;
    if (available_workspace < workspace_size_ || !workspace) {
      throw std::invalid_argument(
          "Cambricon FlashAttnVarlenFunc workspace is insufficient");
    }

    static const std::size_t rng_state[2]{0, 0};
    INFINI_OPS_CNNL_CHECK(cnnlFlashAttentionForward_v2(
        cnnl_handle_.get(), flash_desc_.get(), q_desc_.get(), q.data(),
        k_desc_.get(), k.data(), v_desc_.get(), v.data(), cu_q_desc_.get(),
        cu_seqlens_q.data(), cu_k_desc_.get(), cu_seqlens_k.data(), nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, rng_state, workspace,
        workspace_size_, nullptr, nullptr, nullptr, nullptr, out_desc_.get(),
        out.data()));
  }

  std::size_t workspace_size_in_bytes() const override {
    return workspace_size_;
  }

 private:
  float scale_{1.0F};
  std::size_t workspace_size_{0};
  cnrt_utils::DeviceBuffer default_workspace_{};
  cnnl_utils::Handle cnnl_handle_{};
  cambricon_flash_attn_detail::FlashAttentionDescriptor flash_desc_{};
  cnnl_utils::TensorDescriptor q_desc_{};
  cnnl_utils::TensorDescriptor k_desc_{};
  cnnl_utils::TensorDescriptor v_desc_{};
  cnnl_utils::TensorDescriptor cu_q_desc_{};
  cnnl_utils::TensorDescriptor cu_k_desc_{};
  cnnl_utils::TensorDescriptor out_desc_{};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CAMBRICON_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_
