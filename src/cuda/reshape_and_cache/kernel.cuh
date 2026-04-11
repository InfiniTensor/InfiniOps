#ifndef INFINI_OPS_CUDA_RESHAPE_AND_CACHE_KERNEL_CUH_
#define INFINI_OPS_CUDA_RESHAPE_AND_CACHE_KERNEL_CUH_

#include <cstddef>
#include <cstdint>

namespace infini::ops {

// Writes key and value tensors into a paged KV cache using a slot mapping.
//
// Each thread block processes one token. Threads within the block cooperatively
// write all (num_kv_heads * head_size) elements for that token into both the
// key cache and value cache.
//
// KV cache layout: [2, num_blocks, block_size, num_kv_heads, head_size]
//   - Index 0 along dim 0 is the key cache.
//   - Index 1 along dim 0 is the value cache.
//
// Key/value layout: [num_tokens, num_kv_heads, head_size]
//
// Slot mapping: [num_tokens] — maps each token to a flat slot index in the
// cache. `block_idx = slot / block_size`, `block_offset = slot % block_size`.
template <typename T, unsigned int BLOCK_SIZE>
__global__ void ReshapeAndCacheKernel(
    const T* __restrict__ key, const T* __restrict__ value,
    T* __restrict__ kv_cache_out, const int64_t* __restrict__ slot_mapping,
    size_t num_kv_heads, size_t head_size, size_t block_size,
    size_t num_blocks) {
  const size_t token_idx = blockIdx.x;
  const int64_t slot = slot_mapping[token_idx];

  // Padding tokens have slot_mapping == -1; skip them.
  if (slot < 0) {
    return;
  }

  const size_t block_idx = static_cast<size_t>(slot) / block_size;
  const size_t block_offset = static_cast<size_t>(slot) % block_size;

  const size_t elems_per_token = num_kv_heads * head_size;

  // Compute base offsets into the contiguous KV cache.
  // Cache shape: [2, num_blocks, block_size, num_kv_heads, head_size]
  // Strides:     [num_blocks*block_size*num_kv_heads*head_size,
  //               block_size*num_kv_heads*head_size,
  //               num_kv_heads*head_size,
  //               head_size,
  //               1]
  const size_t cache_block_stride = block_size * num_kv_heads * head_size;
  const size_t cache_kv_stride = num_blocks * cache_block_stride;

  const size_t key_cache_base =
      block_idx * cache_block_stride + block_offset * num_kv_heads * head_size;
  const size_t value_cache_base = cache_kv_stride + key_cache_base;

  // Source offset for this token: key/value shape is [num_tokens, num_kv_heads,
  // head_size], contiguous.
  const size_t src_base = token_idx * elems_per_token;

  for (size_t i = threadIdx.x; i < elems_per_token; i += BLOCK_SIZE) {
    const T k = key[src_base + i];
    const T v = value[src_base + i];

    kv_cache_out[key_cache_base + i] = k;
    kv_cache_out[value_cache_base + i] = v;
  }
}

}  // namespace infini::ops

#endif
