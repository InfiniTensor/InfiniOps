#ifndef INFINI_OPS_CUDA_PAGED_GATHER_KERNEL_CUH_
#define INFINI_OPS_CUDA_PAGED_GATHER_KERNEL_CUH_

#include <cstddef>
#include <cstdint>

namespace infini::ops {

// Scatters values from a paged `k_cache` / `v_cache` into contiguous dense
// buffers. One CUDA block per `(head, position)`; threads cooperatively copy
// `head_size` elements along the last dim.
//
// For a single sequence with effective length `seqlen_k`:
//   `dense_k[pos, h, d] = k_cache[block_table[pos / block_size], h, pos %
//   block_size, d]`.
//
// Cache strides are passed explicitly so the kernel can follow the
// post-`permute` layout `[num_blocks, block_size, num_kv_heads, head_size]`
// that InfiniLM hands us without first materializing a contiguous copy.
template <typename TData, typename TBlockIdx, int BLOCK_SIZE>
__global__ void PagedGatherKernel(
    TData *__restrict__ dense_k, TData *__restrict__ dense_v,
    const TData *__restrict__ k_cache, const TData *__restrict__ v_cache,
    const TBlockIdx *__restrict__ block_table, size_t block_size,
    size_t head_size, std::ptrdiff_t k_cache_block_stride,
    std::ptrdiff_t v_cache_block_stride,
    std::ptrdiff_t k_cache_slot_stride,
    std::ptrdiff_t v_cache_slot_stride,
    std::ptrdiff_t k_cache_head_stride,
    std::ptrdiff_t v_cache_head_stride,
    std::ptrdiff_t dense_seqlen_stride,
    std::ptrdiff_t dense_head_stride) {
  const int head_idx = blockIdx.x;
  const int pos = blockIdx.y;

  const int block_num = pos / static_cast<int>(block_size);
  const int within = pos - block_num * static_cast<int>(block_size);
  const std::int64_t phys_block =
      static_cast<std::int64_t>(block_table[block_num]);

  const TData *k_src = k_cache + phys_block * k_cache_block_stride +
                       static_cast<std::int64_t>(within) * k_cache_slot_stride +
                       static_cast<std::int64_t>(head_idx) * k_cache_head_stride;
  const TData *v_src = v_cache + phys_block * v_cache_block_stride +
                       static_cast<std::int64_t>(within) * v_cache_slot_stride +
                       static_cast<std::int64_t>(head_idx) * v_cache_head_stride;
  TData *k_dst =
      dense_k + static_cast<std::int64_t>(pos) * dense_seqlen_stride +
      static_cast<std::int64_t>(head_idx) * dense_head_stride;
  TData *v_dst =
      dense_v + static_cast<std::int64_t>(pos) * dense_seqlen_stride +
      static_cast<std::int64_t>(head_idx) * dense_head_stride;

  for (size_t i = threadIdx.x; i < head_size; i += BLOCK_SIZE) {
    k_dst[i] = k_src[i];
    v_dst[i] = v_src[i];
  }
}

}  // namespace infini::ops

#endif
