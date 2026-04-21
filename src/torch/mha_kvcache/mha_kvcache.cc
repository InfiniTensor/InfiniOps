#include "torch/mha_kvcache/mha_kvcache.h"

#include "torch/tensor_.h"

#include <ATen/TensorIndexing.h>
#include <ATen/ops/scaled_dot_product_attention.h>

#include <limits>

namespace infini::ops {

template <Device::Type kDev>
Operator<MhaKvcache, kDev, 1>::Operator(const Tensor q, const Tensor k_cache,
                                        const Tensor v_cache,
                                        const Tensor seqlens_k,
                                        const Tensor block_table, float scale,
                                        Tensor out)
    : MhaKvcache{q,         k_cache,     v_cache, seqlens_k,
                 block_table, scale,     out},
      device_index_{out.device().index()} {}

template <Device::Type kDev>
void Operator<MhaKvcache, kDev, 1>::operator()(
    const Tensor q, const Tensor k_cache, const Tensor v_cache,
    const Tensor seqlens_k, const Tensor block_table, float scale,
    Tensor out) const {
  auto at_q = ToAtenTensor<kDev>(const_cast<void*>(q.data()), q_shape_,
                                 q_strides_, q_type_, device_index_);
  auto at_k_cache =
      ToAtenTensor<kDev>(const_cast<void*>(k_cache.data()), k_cache_shape_,
                         k_cache_strides_, k_cache_type_, device_index_);
  auto at_v_cache =
      ToAtenTensor<kDev>(const_cast<void*>(v_cache.data()), v_cache_shape_,
                         v_cache_strides_, v_cache_type_, device_index_);
  auto at_seqlens_k = ToAtenTensor<kDev>(const_cast<void*>(seqlens_k.data()),
                                         seqlens_k_shape_, seqlens_k_strides_,
                                         seqlens_k_type_, device_index_);
  auto at_block_table = ToAtenTensor<kDev>(
      const_cast<void*>(block_table.data()), block_table_shape_,
      block_table_strides_, block_table_type_, device_index_);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   out_type_, device_index_);

  const auto batch_size = at_q.size(0);
  const auto block_size = at_k_cache.size(1);

  // Gather the first `max_len` key/value positions from the paged cache into
  // a dense `[B, max_len, H_k, D]` tensor. Positions beyond `seqlens_k[b]`
  // are masked out below, so their contents are irrelevant.
  const auto max_len = at_seqlens_k.max().template item<int64_t>();
  if (max_len == 0) {
    at_out.zero_();
    return;
  }

  auto pos = at::arange(max_len,
                        at::TensorOptions().device(at_q.device()).dtype(at::kLong));
  auto block_idx = pos.div(block_size, "floor");
  auto within = pos.remainder(block_size);
  auto block_idx_b = block_idx.unsqueeze(0).expand({batch_size, max_len});
  auto within_b = within.unsqueeze(0).expand({batch_size, max_len});

  // `block_table` may be narrower than `max_len / block_size`; clamp to
  // avoid out-of-range gathers for padding positions.
  auto block_table_long = at_block_table.to(at::kLong);
  auto clamped_block_idx =
      at::clamp(block_idx_b, 0, at_block_table.size(1) - 1);
  auto phys_blocks = block_table_long.gather(1, clamped_block_idx);

  using at::indexing::Slice;
  auto k = at_k_cache.index({phys_blocks, within_b, Slice(), Slice()});
  auto v = at_v_cache.index({phys_blocks, within_b, Slice(), Slice()});

  // Build a `[B, max_len]` validity mask and promote it to `[B, 1, 1,
  // max_len]` for `scaled_dot_product_attention`.
  auto seqlens_k_long = at_seqlens_k.to(at::kLong);
  auto key_mask =
      at::arange(max_len,
                 at::TensorOptions().device(at_q.device()).dtype(at::kLong))
          .unsqueeze(0)
          .lt(seqlens_k_long.unsqueeze(1));
  auto attn_mask = at::zeros({batch_size, 1, 1, max_len}, at_q.options());
  attn_mask = attn_mask.masked_fill(
      ~key_mask.unsqueeze(1).unsqueeze(1),
      -std::numeric_limits<float>::infinity());

  // `scaled_dot_product_attention` expects `[B, H, S, D]` layout.
  auto q_sdpa = at_q.transpose(1, 2);
  auto k_sdpa = k.transpose(1, 2);
  auto v_sdpa = v.transpose(1, 2);

  auto result = at::scaled_dot_product_attention(
      q_sdpa, k_sdpa, v_sdpa, attn_mask, /*dropout_p=*/0.0,
      /*is_causal=*/false, /*scale=*/static_cast<double>(scale),
      /*enable_gqa=*/true);

  at_out.copy_(result.transpose(1, 2));
}

template class Operator<MhaKvcache, Device::Type::kCpu, 1>;
template class Operator<MhaKvcache, Device::Type::kNvidia, 1>;
template class Operator<MhaKvcache, Device::Type::kCambricon, 1>;
template class Operator<MhaKvcache, Device::Type::kAscend, 1>;
template class Operator<MhaKvcache, Device::Type::kMetax, 1>;
template class Operator<MhaKvcache, Device::Type::kMoore, 1>;
template class Operator<MhaKvcache, Device::Type::kIluvatar, 1>;
template class Operator<MhaKvcache, Device::Type::kKunlun, 1>;
template class Operator<MhaKvcache, Device::Type::kHygon, 1>;
template class Operator<MhaKvcache, Device::Type::kQy, 1>;

}  // namespace infini::ops
