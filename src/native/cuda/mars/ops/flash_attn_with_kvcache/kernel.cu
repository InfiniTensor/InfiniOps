#include "native/cuda/mars/ops/flash_attn_with_kvcache/kernel.h"

namespace infini::ops {

void Operator<FlashAttnWithKvcache, Device::Type::kMars, 0>::operator()(
    const Tensor, Tensor, Tensor, const std::optional<Tensor>,
    const std::optional<Tensor>, const std::optional<Tensor>,
    const std::optional<Tensor>, const int64_t, const std::optional<Tensor>,
    const std::optional<Tensor>, const std::optional<Tensor>,
    const std::optional<Tensor>, const std::optional<double>, const bool,
    const std::vector<int64_t>, const double, const bool, const int64_t,
    const bool, Tensor, std::optional<Tensor>) const {
  throw std::invalid_argument(
      "Mars FlashAttnWithKvcache requires tensor cache_seqlens");
}

void Operator<FlashAttnWithKvcache, Device::Type::kMars, 0>::operator()(
    const Tensor q, Tensor k_cache, Tensor v_cache,
    const std::optional<Tensor> k, const std::optional<Tensor> v,
    const std::optional<Tensor> rotary_cos,
    const std::optional<Tensor> rotary_sin,
    const std::optional<Tensor> cache_seqlens,
    const std::optional<Tensor> cache_batch_idx,
    const std::optional<Tensor> cache_leftpad,
    const std::optional<Tensor> block_table,
    const std::optional<Tensor> alibi_slopes,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool rotary_interleaved, const int64_t num_splits,
    const bool return_softmax_lse, Tensor out,
    std::optional<Tensor> softmax_lse) const {
  if (!cache_seqlens.has_value() || !block_table.has_value())
    throw std::invalid_argument(
        "Mars FlashAttnWithKvcache requires paged cache metadata");
  if (k || v || rotary_cos || rotary_sin || cache_batch_idx || cache_leftpad ||
      alibi_slopes || softmax_lse || return_softmax_lse || softcap != 0.0 ||
      num_splits < 0)
    throw std::invalid_argument(
        "Mars FlashAttnWithKvcache supports inference-only paged attention");
  if (q.dtype() != DataType::kFloat16 && q.dtype() != DataType::kBFloat16)
    throw std::invalid_argument(
        "Mars FlashAttnWithKvcache requires float16 or bfloat16");

  Tensor_t hc_q = mars_flash_attn::MakeHcTensor(q);
  Tensor_t hc_k_cache = mars_flash_attn::MakeHcTensor(k_cache);
  Tensor_t hc_v_cache = mars_flash_attn::MakeHcTensor(v_cache);
  Tensor_t hc_out = mars_flash_attn::MakeHcTensor(out);
  Tensor_t hc_seqlens = mars_flash_attn::MakeHcTensor(*cache_seqlens);
  Tensor_t hc_block_table = mars_flash_attn::MakeHcTensor(*block_table);
  if (!workspace_)
    throw std::invalid_argument(
        "Mars FlashAttnWithKvcache requires a softmax_lse workspace");
  const int batch = static_cast<int>(q.size(0));
  const int heads = static_cast<int>(q.size(2));
  const int query_length = static_cast<int>(q.size(1));
  Tensor_t hc_softmax_lse =
      make_tensor3d(workspace_, HCFLASHATTN_DATATYPE_FP32, batch, heads,
                    query_length, heads * query_length, query_length, 1);
  const float scale = mars_flash_attn::SoftmaxScaleOrDefault(
      softmax_scale, static_cast<std::size_t>(q.size(3)));

  const auto status = mha_fwd_kvcache(
      hc_q, hc_k_cache, hc_v_cache, nullptr, nullptr, hc_seqlens, nullptr,
      nullptr, nullptr, hc_block_table, nullptr, hc_softmax_lse, hc_out, scale,
      causal, static_cast<int>(window_size.at(0)),
      static_cast<int>(window_size.at(1)), rotary_interleaved,
      mars_flash_attn::ToHcStream(stream_ ? stream_ : nullptr), 1, nullptr,
      nullptr, nullptr);

  mars_flash_attn::ReleaseHcTensor(hc_softmax_lse);
  mars_flash_attn::ReleaseHcTensor(hc_block_table);
  mars_flash_attn::ReleaseHcTensor(hc_seqlens);
  mars_flash_attn::ReleaseHcTensor(hc_out);
  mars_flash_attn::ReleaseHcTensor(hc_v_cache);
  mars_flash_attn::ReleaseHcTensor(hc_k_cache);
  mars_flash_attn::ReleaseHcTensor(hc_q);
  mars_flash_attn::CheckHcStatus(status, "mha_fwd_kvcache");
}

}  // namespace infini::ops
