#include "torch/mha_varlen/mha_varlen.h"

#include "torch/tensor_.h"

#include <ATen/TensorIndexing.h>
#include <ATen/ops/scaled_dot_product_attention.h>

#include <limits>

namespace infini::ops {

template <Device::Type kDev>
Operator<MhaVarlen, kDev, 1>::Operator(const Tensor q, const Tensor k_cache,
                                       const Tensor v_cache,
                                       const Tensor cum_seqlens_q,
                                       const Tensor cum_seqlens_k,
                                       const Tensor block_table, float scale,
                                       Tensor out)
    : MhaVarlen{q,           k_cache,     v_cache,      cum_seqlens_q,
                cum_seqlens_k, block_table, scale,       out},
      device_index_{out.device().index()} {}

template <Device::Type kDev>
void Operator<MhaVarlen, kDev, 1>::operator()(
    const Tensor q, const Tensor k_cache, const Tensor v_cache,
    const Tensor cum_seqlens_q, const Tensor cum_seqlens_k,
    const Tensor block_table, float scale, Tensor out) const {
  auto at_q = ToAtenTensor<kDev>(const_cast<void*>(q.data()), q_shape_,
                                 q_strides_, q_type_, device_index_);
  auto at_k_cache =
      ToAtenTensor<kDev>(const_cast<void*>(k_cache.data()), k_cache_shape_,
                         k_cache_strides_, k_cache_type_, device_index_);
  auto at_v_cache =
      ToAtenTensor<kDev>(const_cast<void*>(v_cache.data()), v_cache_shape_,
                         v_cache_strides_, v_cache_type_, device_index_);
  auto at_cum_seqlens_q = ToAtenTensor<kDev>(
      const_cast<void*>(cum_seqlens_q.data()), cum_seqlens_q_shape_,
      cum_seqlens_q_strides_, cum_seqlens_q_type_, device_index_);
  auto at_cum_seqlens_k = ToAtenTensor<kDev>(
      const_cast<void*>(cum_seqlens_k.data()), cum_seqlens_k_shape_,
      cum_seqlens_k_strides_, cum_seqlens_k_type_, device_index_);
  auto at_block_table = ToAtenTensor<kDev>(
      const_cast<void*>(block_table.data()), block_table_shape_,
      block_table_strides_, block_table_type_, device_index_);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   out_type_, device_index_);

  // `cum_seqlens_*` drive a host-side loop over sequences; bring them to the
  // CPU once and read via `int64_t` for arithmetic.
  auto cu_q_cpu = at_cum_seqlens_q.to(at::kCPU).to(at::kLong);
  auto cu_k_cpu = at_cum_seqlens_k.to(at::kCPU).to(at::kLong);
  const auto batch_size = cu_q_cpu.size(0) - 1;
  const auto* cu_q_ptr = cu_q_cpu.template data_ptr<int64_t>();
  const auto* cu_k_ptr = cu_k_cpu.template data_ptr<int64_t>();

  const auto block_size = at_k_cache.size(1);
  auto block_table_long = at_block_table.to(at::kLong);

  using at::indexing::Slice;
  for (int64_t b = 0; b < batch_size; ++b) {
    const auto q_start = cu_q_ptr[b];
    const auto q_end = cu_q_ptr[b + 1];
    const auto k_start = cu_k_ptr[b];
    const auto k_end = cu_k_ptr[b + 1];
    const auto seqlen_q = q_end - q_start;
    const auto seqlen_k = k_end - k_start;

    if (seqlen_q == 0 || seqlen_k == 0) {
      continue;
    }

    // `[seqlen_q, H_q, D]` slice of packed queries for this sequence.
    auto q_b = at_q.slice(0, q_start, q_end);

    // Gather `seqlen_k` key/value rows from the paged cache for sequence `b`.
    auto pos = at::arange(
        seqlen_k,
        at::TensorOptions().device(at_q.device()).dtype(at::kLong));
    auto block_idx = pos.div(block_size, "floor");
    auto within = pos.remainder(block_size);
    auto clamped_block_idx =
        at::clamp(block_idx, 0, at_block_table.size(1) - 1);
    auto phys_blocks = block_table_long.select(0, b).gather(0, clamped_block_idx);

    auto k_b = at_k_cache.index({phys_blocks, within, Slice(), Slice()});
    auto v_b = at_v_cache.index({phys_blocks, within, Slice(), Slice()});

    // `scaled_dot_product_attention` expects `[B, H, S, D]`. Promote this
    // single sequence to batch 1.
    auto q_sdpa = q_b.transpose(0, 1).unsqueeze(0);
    auto k_sdpa = k_b.transpose(0, 1).unsqueeze(0);
    auto v_sdpa = v_b.transpose(0, 1).unsqueeze(0);

    at::Tensor result;
    if (seqlen_k == seqlen_q) {
      result = at::scaled_dot_product_attention(
          q_sdpa, k_sdpa, v_sdpa, /*attn_mask=*/std::nullopt,
          /*dropout_p=*/0.0, /*is_causal=*/true,
          /*scale=*/static_cast<double>(scale),
          /*enable_gqa=*/true);
    } else {
      // Queries align to the end of the key range. Allowed:
      // `i_k <= (seqlen_k - seqlen_q) + i_q`.
      auto rows = at::arange(
          seqlen_q,
          at::TensorOptions().device(at_q.device()).dtype(at::kLong));
      auto cols = at::arange(
          seqlen_k,
          at::TensorOptions().device(at_q.device()).dtype(at::kLong));
      auto allowed =
          cols.unsqueeze(0).le(rows.unsqueeze(1) + (seqlen_k - seqlen_q));
      auto attn_mask = at::zeros({seqlen_q, seqlen_k}, at_q.options());
      attn_mask = attn_mask.masked_fill(
          ~allowed, -std::numeric_limits<float>::infinity());
      auto attn_mask_4d = attn_mask.view({1, 1, seqlen_q, seqlen_k});

      result = at::scaled_dot_product_attention(
          q_sdpa, k_sdpa, v_sdpa, attn_mask_4d, /*dropout_p=*/0.0,
          /*is_causal=*/false, /*scale=*/static_cast<double>(scale),
          /*enable_gqa=*/true);
    }

    // `result`: `[1, H_q, seqlen_q, D]` -> write back as `[seqlen_q, H_q, D]`.
    at_out.slice(0, q_start, q_end)
        .copy_(result.squeeze(0).transpose(0, 1));
  }
}

template class Operator<MhaVarlen, Device::Type::kCpu, 1>;
template class Operator<MhaVarlen, Device::Type::kNvidia, 1>;
template class Operator<MhaVarlen, Device::Type::kCambricon, 1>;
template class Operator<MhaVarlen, Device::Type::kAscend, 1>;
template class Operator<MhaVarlen, Device::Type::kMetax, 1>;
template class Operator<MhaVarlen, Device::Type::kMoore, 1>;
template class Operator<MhaVarlen, Device::Type::kIluvatar, 1>;
template class Operator<MhaVarlen, Device::Type::kKunlun, 1>;
template class Operator<MhaVarlen, Device::Type::kHygon, 1>;
template class Operator<MhaVarlen, Device::Type::kQy, 1>;

}  // namespace infini::ops
