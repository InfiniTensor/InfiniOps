#include "torch/mha_kvcache/mha_kvcache.h"

#include "torch/tensor_.h"

#include <ATen/TensorIndexing.h>
#include <ATen/ops/scaled_dot_product_attention.h>

#include <limits>
#include <vector>

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
  const auto seqlen_q = at_q.size(1);
  const auto num_kv_heads = at_k_cache.size(2);
  const auto head_size = at_k_cache.size(3);
  const auto block_size = at_k_cache.size(1);

  // Read per-sequence KV lengths to the host once (one `cudaMemcpyAsync` +
  // sync). Looping over the batch on the host is cheap — in return, each
  // sequence's SDPA call sees contiguous K/V with no attention mask, which
  // lets ATen dispatch to flash-attention instead of the dense math backend.
  auto seqlens_k_cpu = at_seqlens_k.to(at::kCPU).to(at::kLong);
  const auto* seqlens_k_host = seqlens_k_cpu.template data_ptr<int64_t>();
  auto block_table_long = at_block_table.to(at::kLong);

  using at::indexing::Slice;
  for (int64_t b = 0; b < batch_size; ++b) {
    const auto seqlen_k = seqlens_k_host[b];
    if (seqlen_k == 0) {
      at_out.select(0, b).zero_();
      continue;
    }

    // Gather exactly `seqlen_k` positions for sequence `b` from the paged
    // cache. Shape: `[seqlen_k, H_k, D]`.
    auto pos = at::arange(
        seqlen_k,
        at::TensorOptions().device(at_q.device()).dtype(at::kLong));
    auto block_idx = pos.div(block_size, "floor");
    auto within = pos.remainder(block_size);
    auto clamped_block_idx =
        at::clamp(block_idx, 0, at_block_table.size(1) - 1);
    auto phys_blocks =
        block_table_long.select(0, b).gather(0, clamped_block_idx);

    auto k_b = at_k_cache.index({phys_blocks, within, Slice(), Slice()});
    auto v_b = at_v_cache.index({phys_blocks, within, Slice(), Slice()});

    // `scaled_dot_product_attention` expects `[B, H, S, D]`. Promote this
    // single sequence to batch 1 so flash-attention can be selected.
    auto q_sdpa = at_q.select(0, b).transpose(0, 1).unsqueeze(0);
    auto k_sdpa = k_b.transpose(0, 1).unsqueeze(0).contiguous();
    auto v_sdpa = v_b.transpose(0, 1).unsqueeze(0).contiguous();

    auto result = at::scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, /*attn_mask=*/std::nullopt,
        /*dropout_p=*/0.0, /*is_causal=*/false,
        /*scale=*/static_cast<double>(scale),
        /*enable_gqa=*/true);

    // `result`: `[1, H_q, seqlen_q, D]` -> write as `[seqlen_q, H_q, D]`.
    at_out.select(0, b).copy_(result.squeeze(0).transpose(0, 1));
  }
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
