#include "native/cuda/mars/ops/flash_attn_varlen_func/kernel.h"

namespace infini::ops {

void Operator<FlashAttnVarlenFunc, Device::Type::kMars, 0>::operator()(
    const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
    const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,
    const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
    const int64_t max_seqlen_k, const double dropout_p,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool deterministic, const bool return_attn_probs, Tensor out,
    std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask) const {
  if (!block_table.has_value())
    throw std::invalid_argument(
        "Mars FlashAttnVarlenFunc requires a paged block table");
  if (alibi_slopes || softmax_lse || s_dmask || return_attn_probs ||
      deterministic || dropout_p != 0.0 || softcap != 0.0)
    throw std::invalid_argument(
        "Mars FlashAttnVarlenFunc supports inference only");
  if (q.dtype() != DataType::kFloat16 && q.dtype() != DataType::kBFloat16)
    throw std::invalid_argument(
        "Mars FlashAttnVarlenFunc requires float16 or bfloat16");

  Tensor_t hc_q = mars_flash_attn::MakeHcTensor(q);
  Tensor_t hc_k = mars_flash_attn::MakeHcTensor(k);
  Tensor_t hc_v = mars_flash_attn::MakeHcTensor(v);
  Tensor_t hc_out = mars_flash_attn::MakeHcTensor(out);
  Tensor_t hc_cu_q = mars_flash_attn::MakeHcTensor(cu_seqlens_q);
  Tensor_t hc_cu_k = mars_flash_attn::MakeHcTensor(cu_seqlens_k);
  Tensor_t hc_block_table = mars_flash_attn::MakeHcTensor(*block_table);
  hcflashattnExtendParameter_t extend = make_extend_param();
  set_extend_parameter_block_table(extend,
                                   const_cast<void*>(block_table->data()),
                                   static_cast<int>(block_table->size(0)),
                                   static_cast<int>(block_table->size(1)));

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t total_k = k.size(0) * k.size(1);
  const float scale = mars_flash_attn::SoftmaxScaleOrDefault(
      softmax_scale, static_cast<std::size_t>(q.size(2)));
  const auto status = mha_varlen_fwd(
      batch_size, q.size(0), q.size(1), total_k, k.size(2), q.size(2), hc_q,
      hc_k, hc_v, hc_out, hc_cu_q, hc_cu_k, nullptr, nullptr, nullptr, nullptr,
      nullptr, static_cast<int>(max_seqlen_q), static_cast<int>(max_seqlen_k),
      0.0f, scale, causal, static_cast<int>(window_size.at(0)),
      static_cast<int>(window_size.at(1)),
      mars_flash_attn::ToHcStream(stream_ ? stream_ : nullptr), extend);

  release_extend_param(extend);
  mars_flash_attn::ReleaseHcTensor(hc_block_table);
  mars_flash_attn::ReleaseHcTensor(hc_cu_k);
  mars_flash_attn::ReleaseHcTensor(hc_cu_q);
  mars_flash_attn::ReleaseHcTensor(hc_out);
  mars_flash_attn::ReleaseHcTensor(hc_v);
  mars_flash_attn::ReleaseHcTensor(hc_k);
  mars_flash_attn::ReleaseHcTensor(hc_q);
  mars_flash_attn::CheckHcStatus(status, "mha_varlen_fwd");
}

}  // namespace infini::ops
