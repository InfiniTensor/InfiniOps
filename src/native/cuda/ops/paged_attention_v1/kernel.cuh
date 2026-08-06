#ifndef INFINI_OPS_CUDA_PAGED_ATTENTION_V1_KERNEL_CUH_
#define INFINI_OPS_CUDA_PAGED_ATTENTION_V1_KERNEL_CUH_

#include <cfloat>
#include <cmath>
#include <cstddef>

namespace infini::ops {

template <typename TIndex, typename TData, int kHeadSize>
__global__ void PagedAttentionDecodeWarpKernel(
    TData* __restrict__ out, const TData* __restrict__ q,
    const TData* __restrict__ k_cache, const TData* __restrict__ v_cache,
    const TIndex* __restrict__ block_tables, const TIndex* __restrict__ seqlens,
    const float* __restrict__ alibi_slopes, std::size_t num_heads,
    std::size_t num_kv_heads, float scale, std::size_t max_num_blocks_per_seq,
    std::size_t block_size, std::ptrdiff_t k_cacheblock_stride,
    std::ptrdiff_t k_cachehead_stride, std::ptrdiff_t k_cacheslot_stride,
    std::ptrdiff_t k_cachedim_stride, std::ptrdiff_t k_cachex_stride,
    int k_cachex, std::ptrdiff_t v_cacheblock_stride,
    std::ptrdiff_t v_cachehead_stride, std::ptrdiff_t v_cacheslot_stride,
    std::ptrdiff_t v_cachedim_stride, std::ptrdiff_t qstride,
    std::ptrdiff_t qhead_stride, std::ptrdiff_t outstride,
    std::ptrdiff_t outhead_stride, std::ptrdiff_t block_table_batch_stride,
    std::ptrdiff_t seqlens_stride) {
  constexpr int kWarpSize = 32;
  static_assert(kHeadSize == 64 || kHeadSize == 128,
                "Paged attention decode supports head sizes 64 and 128");
  static_assert(kHeadSize % kWarpSize == 0,
                "head size must be divisible by 32");

  const int seqidx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int lane = threadIdx.x;
  constexpr int kDimsPerThread = kHeadSize / kWarpSize;
  constexpr float kLog2e = 1.4426950408889634f;

  __shared__ float reduce_buf[kWarpSize];
  __shared__ float state_buf[2];

  const int seqlen = static_cast<int>(seqlens[seqidx * seqlens_stride]);
  TData* outptr = out + seqidx * outstride + head_idx * outhead_stride;
  if (seqlen <= 0) {
#pragma unroll
    for (int i = 0; i < kDimsPerThread; ++i) {
      outptr[lane * kDimsPerThread + i] = static_cast<TData>(0.0f);
    }
    return;
  }

  const int queries_per_kv = static_cast<int>(num_heads / num_kv_heads);
  const int kv_head_idx = head_idx / queries_per_kv;
  const float alibi_slope =
      alibi_slopes == nullptr ? 0.0f : alibi_slopes[head_idx];
  const float scale_log2 = scale * kLog2e;
  const TIndex* block_table = block_tables + seqidx * block_table_batch_stride;
  const TData* qptr = q + seqidx * qstride + head_idx * qhead_stride;

  float qreg[kDimsPerThread];
  float acc[kDimsPerThread];
#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    qreg[i] = static_cast<float>(qptr[dim]);
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
    const TData* k_base = k_cache + physical_block * k_cacheblock_stride +
                          kv_head_idx * k_cachehead_stride;
    const TData* v_base = v_cache + physical_block * v_cacheblock_stride +
                          kv_head_idx * v_cachehead_stride;
    const int token_end = min(page_block_size, seqlen - t_base);

    for (int token_in_block = 0; token_in_block < token_end; ++token_in_block) {
      const int token_idx = t_base + token_in_block;
      const TData* k_ptr = k_base + token_in_block * k_cacheslot_stride;
      const TData* v_ptr = v_base + token_in_block * v_cacheslot_stride;

      float qk = 0.0f;
#pragma unroll
      for (int i = 0; i < kDimsPerThread; ++i) {
        const int dim = lane * kDimsPerThread + i;
        const auto k_offset = (dim / k_cachex) * k_cachedim_stride +
                              (dim % k_cachex) * k_cachex_stride;
        qk += qreg[i] * static_cast<float>(k_ptr[k_offset]);
      }

      reduce_buf[lane] = qk;
      __syncthreads();
      for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
          reduce_buf[lane] += reduce_buf[lane + offset];
        }
        __syncthreads();
      }

      float alpha = 1.0f;
      float beta = 0.0f;
      if (lane == 0) {
        float score = reduce_buf[0] * scale_log2;
        if (alibi_slope != 0.0f) {
          score +=
              (alibi_slope * static_cast<float>(token_idx - (seqlen - 1))) *
              kLog2e;
        }
        const float m_new = fmaxf(m, score);
        alpha = exp2f(m - m_new);
        beta = exp2f(score - m_new);
        l = l * alpha + beta;
        m = m_new;
        state_buf[0] = alpha;
        state_buf[1] = beta;
      }
      __syncthreads();
      alpha = state_buf[0];
      beta = state_buf[1];

#pragma unroll
      for (int i = 0; i < kDimsPerThread; ++i) {
        const int dim = lane * kDimsPerThread + i;
        acc[i] = acc[i] * alpha +
                 beta * static_cast<float>(v_ptr[dim * v_cachedim_stride]);
      }
    }
  }

  if (lane == 0) {
    state_buf[0] = 1.0f / (l + 1e-6f);
  }
  __syncthreads();
  const float inv_l = state_buf[0];

#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    outptr[dim] = static_cast<TData>(acc[i] * inv_l);
  }
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_PAGED_ATTENTION_V1_KERNEL_CUH_
