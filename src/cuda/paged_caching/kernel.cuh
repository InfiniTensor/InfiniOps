#ifndef INFINI_OPS_CUDA_PAGED_CACHING_KERNEL_CUH_
#define INFINI_OPS_CUDA_PAGED_CACHING_KERNEL_CUH_

#include <cstddef>
#include <cstdint>

namespace infini::ops {

// Scatters contiguous `(k, v)` tokens into a paged `(k_cache, v_cache)`
// KV cache. One CUDA block per `(head, token)` — threads cooperatively
// copy `head_size` elements along the last dimension.
//
//   `k_cache`, `v_cache`: `[num_blocks, num_kv_heads, block_size, head_size]`.
//   `k`, `v`:             `[num_tokens, num_kv_heads, head_size]`.
//   `slot_mapping`:       `[num_tokens]` (64-bit). Negative entries are padding
//                         tokens and are skipped.
template <typename TData, typename TIndex, int BLOCK_SIZE>
__global__ void PagedCachingKernel(
    TData *__restrict__ k_cache, TData *__restrict__ v_cache,
    const TData *__restrict__ k, const TData *__restrict__ v,
    const TIndex *__restrict__ slot_mapping, size_t head_size,
    size_t block_size, std::ptrdiff_t k_src_stride,
    std::ptrdiff_t v_src_stride, std::ptrdiff_t k_cache_block_stride,
    std::ptrdiff_t v_cache_block_stride,
    std::ptrdiff_t k_cache_head_stride,
    std::ptrdiff_t v_cache_head_stride,
    std::ptrdiff_t k_cache_slot_stride,
    std::ptrdiff_t v_cache_slot_stride) {
  const int head_idx = blockIdx.x;
  const int token_idx = blockIdx.y;

  const std::int64_t slot = static_cast<std::int64_t>(slot_mapping[token_idx]);
  if (slot < 0) {
    return;
  }
  const std::int64_t physical_block =
      slot / static_cast<std::int64_t>(block_size);
  const std::int64_t block_offset =
      slot % static_cast<std::int64_t>(block_size);

  const TData *k_src = k + token_idx * k_src_stride + head_idx * head_size;
  const TData *v_src = v + token_idx * v_src_stride + head_idx * head_size;
  TData *k_dst = k_cache + physical_block * k_cache_block_stride +
                 head_idx * k_cache_head_stride +
                 block_offset * k_cache_slot_stride;
  TData *v_dst = v_cache + physical_block * v_cache_block_stride +
                 head_idx * v_cache_head_stride +
                 block_offset * v_cache_slot_stride;

  for (size_t i = threadIdx.x; i < head_size; i += BLOCK_SIZE) {
    k_dst[i] = k_src[i];
    v_dst[i] = v_src[i];
  }
}

}  // namespace infini::ops

#endif
