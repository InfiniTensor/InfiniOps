#ifndef INFINI_OPS_CUDA_ROTARY_EMBEDDING_KERNEL_CUH_
#define INFINI_OPS_CUDA_ROTARY_EMBEDDING_KERNEL_CUH_

#include <cstddef>
#include <cstdint>

#include "cuda/caster.cuh"
#include "cuda/kernel_commons.cuh"

namespace infini::ops {

// Applies rotary position embeddings to query and key tensors.
//
// Each thread block handles one token. Threads within the block iterate over
// (head, rot_offset) pairs to apply the rotation formula:
//   arr[x_idx] = x * cos - y * sin
//   arr[y_idx] = y * cos + x * sin
//
// Supports two index patterns:
//   - NeoX style: x_idx = rot_offset, y_idx = half_rotary_dim + rot_offset
//   - GPT-J style: x_idx = 2 * rot_offset, y_idx = 2 * rot_offset + 1
template <unsigned int kBlockSize, Device::Type kDev, typename TCompute,
          typename TData>
__global__ void RotaryEmbeddingKernel(
    TData* __restrict__ query_out, TData* __restrict__ key_out,
    const TData* __restrict__ query, const TData* __restrict__ key,
    const TData* __restrict__ cos_sin_cache,
    const int64_t* __restrict__ positions, int64_t num_heads,
    int64_t num_kv_heads, int64_t head_size, int64_t rotary_dim,
    int64_t query_stride_token, int64_t query_stride_head,
    int64_t key_stride_token, int64_t key_stride_head,
    int64_t query_out_stride_token, int64_t query_out_stride_head,
    int64_t key_out_stride_token, int64_t key_out_stride_head,
    bool is_neox_style) {
  int64_t token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  int64_t half_rotary_dim = rotary_dim / 2;

  // Pointer to the cos/sin row for this token's position.
  // Cache layout: [max_seq_len, rotary_dim] where first half is cos, second
  // half is sin.
  const TData* cos_ptr = cos_sin_cache + pos * rotary_dim;
  const TData* sin_ptr = cos_ptr + half_rotary_dim;

  int64_t total_heads = num_heads + num_kv_heads;
  int64_t total_work = total_heads * half_rotary_dim;

  for (int64_t i = threadIdx.x; i < total_work; i += kBlockSize) {
    int64_t head_idx = i / half_rotary_dim;
    int64_t rot_offset = i % half_rotary_dim;

    TCompute cos_val =
        Caster<kDev>::template Cast<TCompute>(cos_ptr[rot_offset]);
    TCompute sin_val =
        Caster<kDev>::template Cast<TCompute>(sin_ptr[rot_offset]);

    int64_t x_idx, y_idx;

    if (is_neox_style) {
      x_idx = rot_offset;
      y_idx = half_rotary_dim + rot_offset;
    } else {
      x_idx = 2 * rot_offset;
      y_idx = 2 * rot_offset + 1;
    }

    if (head_idx < num_heads) {
      // Apply to query.
      const TData* q_in =
          query + token_idx * query_stride_token + head_idx * query_stride_head;
      TData* q_out = query_out + token_idx * query_out_stride_token +
                     head_idx * query_out_stride_head;

      TCompute x = Caster<kDev>::template Cast<TCompute>(q_in[x_idx]);
      TCompute y = Caster<kDev>::template Cast<TCompute>(q_in[y_idx]);
      q_out[x_idx] = Caster<kDev>::template Cast<TData>(x * cos_val - y * sin_val);
      q_out[y_idx] = Caster<kDev>::template Cast<TData>(y * cos_val + x * sin_val);

      // Copy non-rotary dimensions if needed.
      if (rot_offset == 0 && rotary_dim < head_size) {
        for (int64_t d = rotary_dim; d < head_size; ++d) {
          q_out[d] = q_in[d];
        }
      }
    } else {
      // Apply to key.
      int64_t kv_head_idx = head_idx - num_heads;
      const TData* k_in =
          key + token_idx * key_stride_token + kv_head_idx * key_stride_head;
      TData* k_out = key_out + token_idx * key_out_stride_token +
                     kv_head_idx * key_out_stride_head;

      TCompute x = Caster<kDev>::template Cast<TCompute>(k_in[x_idx]);
      TCompute y = Caster<kDev>::template Cast<TCompute>(k_in[y_idx]);
      k_out[x_idx] = Caster<kDev>::template Cast<TData>(x * cos_val - y * sin_val);
      k_out[y_idx] = Caster<kDev>::template Cast<TData>(y * cos_val + x * sin_val);

      // Copy non-rotary dimensions if needed.
      if (rot_offset == 0 && rotary_dim < head_size) {
        for (int64_t d = rotary_dim; d < head_size; ++d) {
          k_out[d] = k_in[d];
        }
      }
    }
  }
}

}  // namespace infini::ops

#endif
