#include "torch/ops/flash_attn_varlen_func/flash_attn_varlen_func.h"

#include <ATen/ops/_flash_attention_forward.h>

#include <tuple>

#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
void AtenFlashAttnVarlenFunc<kDev>::operator()(
    const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
    const Tensor cu_seqlens_k, const int64_t max_seqlen_q,
    const int64_t max_seqlen_k, const double dropout_p,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const std::optional<Tensor> alibi_slopes, const bool deterministic,
    const bool return_attn_probs, const std::optional<Tensor> block_table,
    Tensor out) const {
  (void)softcap;
  (void)alibi_slopes;
  (void)deterministic;
  (void)return_attn_probs;
  (void)block_table;

  auto at_q = ToAtenTensor<kDev>(const_cast<void*>(q.data()), q_shape_,
                                 q_strides_, q_dtype_, device_index_);
  auto at_k = ToAtenTensor<kDev>(const_cast<void*>(k.data()), k_shape_,
                                 k_strides_, k_dtype_, device_index_);
  auto at_v = ToAtenTensor<kDev>(const_cast<void*>(v.data()), v_shape_,
                                 v_strides_, v_dtype_, device_index_);
  auto at_cu_seqlens_q = ToAtenTensor<kDev>(
      const_cast<void*>(cu_seqlens_q.data()), cu_seqlens_q_shape_,
      cu_seqlens_q_strides_, cu_seqlens_q_dtype_, device_index_);
  auto at_cu_seqlens_k = ToAtenTensor<kDev>(
      const_cast<void*>(cu_seqlens_k.data()), cu_seqlens_k_shape_,
      cu_seqlens_k_strides_, cu_seqlens_k_dtype_, device_index_);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   out_dtype_, device_index_);

  std::optional<int64_t> window_size_left;
  std::optional<int64_t> window_size_right;
  if (window_size[0] >= 0) {
    window_size_left = window_size[0];
  }
  if (causal) {
    window_size_right = 0;
  } else if (window_size[1] >= 0) {
    window_size_right = window_size[1];
  }

  auto result = at::_flash_attention_forward(
      at_q, at_k, at_v, at_cu_seqlens_q, at_cu_seqlens_k, max_seqlen_q,
      max_seqlen_k, dropout_p, causal, false, softmax_scale, window_size_left,
      window_size_right, std::nullopt, std::nullopt);

  // ATen owns the returned tensor. Keep the InfiniOps trailing-output ABI by
  // copying it into the caller-provided buffer on the current ATen stream.
  at_out.copy_(std::get<0>(result));
}

template class AtenFlashAttnVarlenFunc<Device::Type::kNvidia>;
template class AtenFlashAttnVarlenFunc<Device::Type::kMoore>;

}  // namespace infini::ops
