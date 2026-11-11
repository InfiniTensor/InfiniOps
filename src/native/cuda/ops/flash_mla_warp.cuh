#ifndef INFINI_OPS_CUDA_FLASH_MLA_WARP_CUH_
#define INFINI_OPS_CUDA_FLASH_MLA_WARP_CUH_

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "native/cuda/caster.cuh"

namespace op::flash::cuda {

// MLA-packed paged attention over a fused latent KV cache: q/k head dim
// kv_lora_rank + rope (576 for 021) with a separate value latent
// (kv_lora_rank = 512). One warp per (token, head); the query reduction
// spans the packed qk dims while the value accumulation and output span
// the value latent dims. Numerics follow the verified 021 kernels.
constexpr int kWarpSize = 32;
constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ float MlaWarpReduceSum(float x) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, offset);
  }
  return x;
}

__device__ __forceinline__ float MlaWarpBroadcast(float x, int src_lane) {
  return __shfl_sync(0xffffffff, x, src_lane);
}

// Decode: single-token queries, one sequence per block row (batch = q.size(0)).
template <typename TIndex, typename TData, int kQkHeadSize, int kVHeadSize>
__global__ void FlashMlaDecodeWarpKernel(
    TData* __restrict__ out, const TData* __restrict__ q,
    const TData* __restrict__ k_cache, const TData* __restrict__ v_cache,
    const TIndex* __restrict__ block_tables, const TIndex* __restrict__ seqlens,
    std::size_t num_heads, std::size_t num_kv_heads, float scale,
    std::size_t max_num_blocks_per_seq, std::size_t block_size,
    std::ptrdiff_t q_batch_stride, std::ptrdiff_t q_head_stride,
    std::ptrdiff_t k_cache_block_stride, std::ptrdiff_t k_cache_head_stride,
    std::ptrdiff_t k_cache_slot_stride, std::ptrdiff_t v_cache_block_stride,
    std::ptrdiff_t v_cache_head_stride, std::ptrdiff_t v_cache_slot_stride,
    std::ptrdiff_t o_batch_stride, std::ptrdiff_t o_head_stride,
    std::ptrdiff_t block_table_batch_stride, std::ptrdiff_t seqlens_stride) {
  constexpr int kQDimsPerThread = kQkHeadSize / kWarpSize;
  constexpr int kVDimsPerThread = kVHeadSize / kWarpSize;

  const int seqidx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int lane = threadIdx.x;

  const int seqlen = static_cast<int>(seqlens[seqidx * seqlens_stride]);
  TData* outptr = out + seqidx * o_batch_stride + head_idx * o_head_stride;
  if (seqlen <= 0) {
#pragma unroll
    for (int i = 0; i < kVDimsPerThread; ++i) {
      outptr[lane * kVDimsPerThread + i] = static_cast<TData>(0.0f);
    }
    return;
  }

  const int queries_per_kv = static_cast<int>(num_heads / num_kv_heads);
  const int kv_head_idx = head_idx / queries_per_kv;
  const float scale_log2 = scale * kLog2e;
  const TIndex* block_table = block_tables + seqidx * block_table_batch_stride;
  const TData* qptr = q + seqidx * q_batch_stride + head_idx * q_head_stride;

  float qreg[kQDimsPerThread];
  float acc[kVDimsPerThread];
#pragma unroll
  for (int i = 0; i < kQDimsPerThread; ++i) {
    const int dim = lane * kQDimsPerThread + i;
    qreg[i] = static_cast<float>(qptr[dim]);
  }
#pragma unroll
  for (int i = 0; i < kVDimsPerThread; ++i) {
    acc[i] = 0.0f;
  }

  float m = -FLT_MAX;
  float l = 0.0f;
  const int page_block_size = static_cast<int>(block_size);
  int t_base = 0;
  for (int logical_block = 0;
       t_base < seqlen &&
       logical_block < static_cast<int>(max_num_blocks_per_seq);
       ++logical_block, t_base += page_block_size) {
    const int physical_block = static_cast<int>(block_table[logical_block]);
    const TData* k_base = k_cache + physical_block * k_cache_block_stride +
                          kv_head_idx * k_cache_head_stride;
    const TData* v_base = v_cache + physical_block * v_cache_block_stride +
                          kv_head_idx * v_cache_head_stride;
    const int token_end = min(page_block_size, seqlen - t_base);

    for (int token_in_block = 0; token_in_block < token_end; ++token_in_block) {
      const TData* k_ptr = k_base + token_in_block * k_cache_slot_stride;
      const TData* v_ptr = v_base + token_in_block * v_cache_slot_stride;

      float qk = 0.0f;
#pragma unroll
      for (int i = 0; i < kQDimsPerThread; ++i) {
        const int dim = lane * kQDimsPerThread + i;
        qk += qreg[i] * static_cast<float>(k_ptr[dim]);
      }

      qk = MlaWarpReduceSum(qk);

      float alpha = 1.0f;
      float beta = 0.0f;
      if (lane == 0) {
        const float score = qk * scale_log2;
        const float m_new = fmaxf(m, score);
        alpha = exp2f(m - m_new);
        beta = exp2f(score - m_new);
        l = l * alpha + beta;
        m = m_new;
      }
      alpha = MlaWarpBroadcast(alpha, 0);
      beta = MlaWarpBroadcast(beta, 0);

#pragma unroll
      for (int i = 0; i < kVDimsPerThread; ++i) {
        const int dim = lane * kVDimsPerThread + i;
        acc[i] = acc[i] * alpha + beta * static_cast<float>(v_ptr[dim]);
      }
    }
  }

  float inv_l = 0.0f;
  if (lane == 0) {
    inv_l = 1.0f / (l + 1e-6f);
  }
  inv_l = MlaWarpBroadcast(inv_l, 0);

#pragma unroll
  for (int i = 0; i < kVDimsPerThread; ++i) {
    const int dim = lane * kVDimsPerThread + i;
    outptr[dim] = static_cast<TData>(acc[i] * inv_l);
  }
}

// Prefill: packed variable-length queries over a paged latent KV cache.
// cu_seqlens_q carries per-sequence query token offsets and cu_seqlens_k
// the cumulative KV lengths (query included), matching the base interface.
// Causal within each sequence.
template <typename TIndex, typename TData, int kQkHeadSize, int kVHeadSize>
__global__ void FlashMlaPrefillWarpKernel(
    TData* __restrict__ out, const TData* __restrict__ q,
    const TData* __restrict__ k_cache, const TData* __restrict__ v_cache,
    const TIndex* __restrict__ block_tables,
    const TIndex* __restrict__ cu_seqlens_k,
    const TIndex* __restrict__ cu_seqlens_q, std::size_t num_heads,
    std::size_t num_seqs, std::size_t num_kv_heads, std::size_t total_qtokens,
    float scale, std::size_t max_num_blocks_per_seq,
    std::size_t page_block_size, std::ptrdiff_t block_table_batch_stride,
    std::ptrdiff_t q_stride, std::ptrdiff_t q_head_stride,
    std::ptrdiff_t k_batch_stride, std::ptrdiff_t k_row_stride,
    std::ptrdiff_t k_head_stride, std::ptrdiff_t v_batch_stride,
    std::ptrdiff_t v_row_stride, std::ptrdiff_t v_head_stride,
    std::ptrdiff_t o_stride, std::ptrdiff_t o_head_stride) {
  constexpr int kQDimsPerThread = kQkHeadSize / kWarpSize;
  constexpr int kVDimsPerThread = kVHeadSize / kWarpSize;

  const int lane = threadIdx.x;
  const std::size_t head_idx = static_cast<std::size_t>(blockIdx.x);
  const std::size_t global_token_idx = static_cast<std::size_t>(blockIdx.y);

  if (lane >= kWarpSize || head_idx >= num_heads ||
      global_token_idx >= total_qtokens) {
    return;
  }

  std::size_t seq_low = 0;
  std::size_t seq_high = num_seqs;
  while (seq_low + 1 < seq_high) {
    const std::size_t mid = (seq_low + seq_high) >> 1;
    if (static_cast<std::size_t>(cu_seqlens_q[mid]) <= global_token_idx) {
      seq_low = mid;
    } else {
      seq_high = mid;
    }
  }
  const std::size_t seq_idx = seq_low;
  const TIndex qstart = cu_seqlens_q[seq_idx];
  const TIndex qend = cu_seqlens_q[seq_idx + 1];
  const int qlen = static_cast<int>(qend - qstart);

  const int qtoken_local =
      static_cast<int>(global_token_idx - static_cast<std::size_t>(qstart));
  if (qtoken_local < 0 || qtoken_local >= qlen) {
    return;
  }

  const int kv_len_total =
      static_cast<int>(cu_seqlens_k[seq_idx + 1] - cu_seqlens_k[seq_idx]);
  const int history_len = kv_len_total - qlen;
  const int causal_limit = history_len + qtoken_local;
  const int allowed_k_len = causal_limit + 1;

  const int num_queries_per_kv = static_cast<int>(num_heads / num_kv_heads);
  const int kv_head_idx = static_cast<int>(head_idx) / num_queries_per_kv;
  const float scale_log2 = scale * kLog2e;

  const TData* q_ptr = q + static_cast<int64_t>(global_token_idx) * q_stride +
                       static_cast<int64_t>(head_idx) * q_head_stride;
  TData* out_ptr = out + static_cast<int64_t>(global_token_idx) * o_stride +
                   static_cast<int64_t>(head_idx) * o_head_stride;
  const TIndex* block_table =
      block_tables + static_cast<int64_t>(seq_idx) *
                         static_cast<int64_t>(block_table_batch_stride);
  const int pbs = static_cast<int>(page_block_size);

  float q_reg[kQDimsPerThread];
  float acc[kVDimsPerThread];
#pragma unroll
  for (int i = 0; i < kQDimsPerThread; ++i) {
    const int dim = lane * kQDimsPerThread + i;
    q_reg[i] = static_cast<float>(q_ptr[dim]);
  }
#pragma unroll
  for (int i = 0; i < kVDimsPerThread; ++i) {
    acc[i] = 0.0f;
  }

  float m = -INFINITY;
  float l = 0.0f;
  int t_base = 0;
  for (int logical_block = 0;
       t_base < allowed_k_len &&
       logical_block < static_cast<int>(max_num_blocks_per_seq);
       ++logical_block, t_base += pbs) {
    const int physical_block = static_cast<int>(block_table[logical_block]);
    const TData* k_base =
        k_cache + static_cast<int64_t>(physical_block) * k_batch_stride +
        static_cast<int64_t>(kv_head_idx) * k_head_stride;
    const TData* v_base =
        v_cache + static_cast<int64_t>(physical_block) * v_batch_stride +
        static_cast<int64_t>(kv_head_idx) * v_head_stride;
    const int token_end = min(pbs, allowed_k_len - t_base);
    for (int token_in_block = 0; token_in_block < token_end; ++token_in_block) {
      const int t = t_base + token_in_block;
      const TData* k_ptr =
          k_base + static_cast<int64_t>(token_in_block) * k_row_stride;
      const TData* v_ptr =
          v_base + static_cast<int64_t>(token_in_block) * v_row_stride;

      float qk = 0.0f;
#pragma unroll
      for (int i = 0; i < kQDimsPerThread; ++i) {
        const int dim = lane * kQDimsPerThread + i;
        qk += q_reg[i] * static_cast<float>(k_ptr[dim]);
      }
      qk = MlaWarpReduceSum(qk);

      float alpha = 1.0f;
      float beta = 0.0f;
      if (lane == 0) {
        const float score = qk * scale_log2;
        const float m_new = fmaxf(m, score);
        alpha = exp2f(m - m_new);
        beta = exp2f(score - m_new);
        l = l * alpha + beta;
        m = m_new;
      }
      alpha = MlaWarpBroadcast(alpha, 0);
      beta = MlaWarpBroadcast(beta, 0);

#pragma unroll
      for (int i = 0; i < kVDimsPerThread; ++i) {
        const int dim = lane * kVDimsPerThread + i;
        acc[i] = acc[i] * alpha + beta * static_cast<float>(v_ptr[dim]);
      }
    }
  }

  float inv_l = 0.0f;
  if (lane == 0) {
    inv_l = 1.0f / (l + 1e-6f);
  }
  inv_l = MlaWarpBroadcast(inv_l, 0);

#pragma unroll
  for (int i = 0; i < kVDimsPerThread; ++i) {
    const int dim = lane * kVDimsPerThread + i;
    out_ptr[dim] = static_cast<TData>(acc[i] * inv_l);
  }
}

}  // namespace op::flash::cuda

#endif
