#include "torch/mha_kvcache/mha_kvcache.h"

#include "torch/tensor_.h"

#include <ATen/ops/empty.h>
#include <ATen/ops/scaled_dot_product_attention.h>

#include <cstddef>

#if defined(WITH_NVIDIA)
#include <c10/cuda/CUDAStream.h>

#include "cuda/nvidia/paged_gather_launcher.h"
#endif

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
  auto at_seqlens_k = ToAtenTensor<kDev>(const_cast<void*>(seqlens_k.data()),
                                         seqlens_k_shape_, seqlens_k_strides_,
                                         seqlens_k_type_, device_index_);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   out_type_, device_index_);

  const auto batch_size = static_cast<int64_t>(at_q.size(0));
  const auto num_kv_heads = static_cast<int64_t>(k_cache_shape_[2]);
  const auto head_size = static_cast<int64_t>(k_cache_shape_[3]);
  const auto block_size = static_cast<int64_t>(k_cache_shape_[1]);

  auto seqlens_k_cpu = at_seqlens_k.to(at::kCPU).to(at::kLong);
  const auto* seqlens_k_host = seqlens_k_cpu.template data_ptr<int64_t>();

  void *stream = nullptr;
#if defined(WITH_NVIDIA)
  if constexpr (kDev == Device::Type::kNvidia) {
    stream = static_cast<void *>(
        c10::cuda::getCurrentCUDAStream(device_index_).stream());
  }
#endif

  for (int64_t b = 0; b < batch_size; ++b) {
    const auto seqlen_k = seqlens_k_host[b];
    if (seqlen_k == 0) {
      at_out.select(0, b).zero_();
      continue;
    }

    // Allocate a dense `[seqlen_k, num_kv_heads, head_size]` scratch for
    // both `k` and `v` — ATen's allocator is pooled so this is near-free.
    auto k_b = at::empty({seqlen_k, num_kv_heads, head_size}, at_q.options());
    auto v_b = at::empty({seqlen_k, num_kv_heads, head_size}, at_q.options());

#if defined(WITH_NVIDIA)
    if constexpr (kDev == Device::Type::kNvidia) {
      // Native CUDA paged gather: one launch replaces the
      // `arange / div_floor / remainder / gather / advanced-index` ATen
      // chain that previously cost ~150 us per call.
      const auto *k_cache_ptr = reinterpret_cast<const std::byte *>(
          k_cache.data());
      const auto *v_cache_ptr = reinterpret_cast<const std::byte *>(
          v_cache.data());
      const auto *block_table_row_ptr = reinterpret_cast<const std::byte *>(
          block_table.data()) +
          static_cast<std::ptrdiff_t>(block_table_strides_[0]) * b *
              static_cast<std::ptrdiff_t>(kDataTypeToSize.at(block_table_type_));
      LaunchPagedGatherNvidia(
          k_b.data_ptr(), v_b.data_ptr(), k_cache_ptr, v_cache_ptr,
          block_table_row_ptr, k_cache_type_, block_table_type_,
          static_cast<std::size_t>(seqlen_k),
          static_cast<std::size_t>(num_kv_heads),
          static_cast<std::size_t>(head_size),
          static_cast<std::size_t>(block_size), k_cache_strides_[0],
          v_cache_strides_[0], k_cache_strides_[1], v_cache_strides_[1],
          k_cache_strides_[2], v_cache_strides_[2],
          static_cast<std::ptrdiff_t>(num_kv_heads) * head_size, head_size,
          stream);
    } else
#endif
    {
      // Fallback for non-NVIDIA devices: the original ATen gather chain.
      auto at_k_cache = ToAtenTensor<kDev>(const_cast<void*>(k_cache.data()),
                                           k_cache_shape_, k_cache_strides_,
                                           k_cache_type_, device_index_);
      auto at_v_cache = ToAtenTensor<kDev>(const_cast<void*>(v_cache.data()),
                                           v_cache_shape_, v_cache_strides_,
                                           v_cache_type_, device_index_);
      auto at_block_table = ToAtenTensor<kDev>(
          const_cast<void*>(block_table.data()), block_table_shape_,
          block_table_strides_, block_table_type_, device_index_);
      auto pos = at::arange(
          seqlen_k,
          at::TensorOptions().device(at_q.device()).dtype(at::kLong));
      auto block_idx = pos.div(block_size, "floor");
      auto within = pos.remainder(block_size);
      auto clamped_block_idx =
          at::clamp(block_idx, 0, at_block_table.size(1) - 1);
      auto phys_blocks = at_block_table.to(at::kLong).select(0, b).gather(
          0, clamped_block_idx);
      using at::indexing::Slice;
      k_b = at_k_cache.index({phys_blocks, within, Slice(), Slice()});
      v_b = at_v_cache.index({phys_blocks, within, Slice(), Slice()});
    }

    // `scaled_dot_product_attention` wants `[B, H, S, D]` contiguous so
    // flash-attention can be selected.
    auto q_sdpa = at_q.select(0, b).transpose(0, 1).unsqueeze(0);
    auto k_sdpa = k_b.transpose(0, 1).unsqueeze(0).contiguous();
    auto v_sdpa = v_b.transpose(0, 1).unsqueeze(0).contiguous();

    auto result = at::scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, /*attn_mask=*/std::nullopt,
        /*dropout_p=*/0.0, /*is_causal=*/false,
        /*scale=*/static_cast<double>(scale),
        /*enable_gqa=*/true);

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
