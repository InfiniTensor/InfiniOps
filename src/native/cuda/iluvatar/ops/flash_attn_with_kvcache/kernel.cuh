#ifndef INFINI_OPS_ILUVATAR_FLASH_DECODE_KERNEL_CUH_
#define INFINI_OPS_ILUVATAR_FLASH_DECODE_KERNEL_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "native/cuda/iluvatar/ops/flash_attn_with_kvcache/reduction_policy.h"

namespace infini::ops::flash_attn_with_kvcache_iluvatar_detail {

struct DecodeQkParams {
  DecodeQkConfig config;
  const __nv_bfloat16* q;
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  const int32_t* block_table;
  const int32_t* lengths;
  __nv_bfloat16* out;
  float* partial_acc;
  float* partial_max;
  float* partial_sum;
  float scale;
  int splits;
};

template <int kHeadSize>
__global__ void DecodeSplitQkKernel(DecodeQkParams params) {
  static_assert(kHeadSize == 128);
  constexpr int kThreads = 64;
  constexpr int kTokens = 8;
  constexpr int kChunks = kHeadSize / 8;
  constexpr float kLog2e = 1.4426950408889634f;
  const auto& config = params.config;
  const int tid = threadIdx.x;
  const int sequence = blockIdx.y;
  const int head = blockIdx.x;
  const int split = blockIdx.z;
  const int dim = tid * 2;
  const int page_size = static_cast<int>(config.page_size);
  const int capacity =
      static_cast<int>(config.blocks_per_sequence * config.page_size);
  const int length = min(params.lengths[sequence], capacity);
  const int shard =
      length > 0 ? (length + params.splits - 1) / params.splits : 0;
  const int start = split * shard;
  const int end = min(length, start + shard);
  const int rows = gridDim.x * gridDim.y;
  const int row = split * rows + sequence * gridDim.x + head;
  if (start >= end) {
    if (tid == 0) {
      params.partial_max[row] = -INFINITY;
      params.partial_sum[row] = 0.0f;
    }
    params.partial_acc[row * kHeadSize + dim] = 0.0f;
    params.partial_acc[row * kHeadSize + dim + 1] = 0.0f;
    return;
  }

  const int kv_head = head / (config.heads / config.kv_heads);
  const auto* query = params.q + sequence * config.q_batch_stride +
                      head * config.q_head_stride + dim;
  const float2 q =
      __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(query));
  const auto* table =
      params.block_table + sequence * config.blocks_per_sequence;
  const float scale = params.scale * kLog2e;
  float acc0 = 0.0f, acc1 = 0.0f;
  float maximum = -INFINITY, sum = 0.0f;
  __shared__ __align__(16) __nv_bfloat16 keys[kTokens][kHeadSize];
  __shared__ __align__(16) __nv_bfloat16 values[kTokens][kHeadSize];
  __shared__ float reductions[kTokens][kThreads];
  __shared__ float scores[32];
  __shared__ float weights[kTokens];
  __shared__ float alpha;

  for (int logical_block = start / page_size;
       logical_block <= (end - 1) / page_size; ++logical_block) {
    const int physical_block = table[logical_block];
    const auto* key = params.k + physical_block * config.k_block_stride +
                      kv_head * config.k_head_stride;
    const auto* value = params.v + physical_block * config.v_block_stride +
                        kv_head * config.v_head_stride;
    const int page_start = logical_block * page_size;
    const int begin = max(0, start - page_start);
    const int limit = min(page_size, end - page_start);
    for (int token = begin; token < limit; token += kTokens) {
      const int count = min(kTokens, limit - token);
      for (int index = tid; index < kTokens * kChunks; index += kThreads) {
        const int j = index / kChunks;
        const int offset = (index % kChunks) * 8;
        auto* key_dst = reinterpret_cast<uint4*>(&keys[j][offset]);
        auto* value_dst = reinterpret_cast<uint4*>(&values[j][offset]);
        if (j < count) {
          *key_dst = *reinterpret_cast<const uint4*>(
              key + (token + j) * config.k_token_stride + offset);
          *value_dst = *reinterpret_cast<const uint4*>(
              value + (token + j) * config.v_token_stride + offset);
        } else {
          *key_dst = make_uint4(0, 0, 0, 0);
          *value_dst = make_uint4(0, 0, 0, 0);
        }
      }
      __syncthreads();
#pragma unroll
      for (int j = 0; j < kTokens; ++j) {
        const float2 k = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(&keys[j][dim]));
        reductions[j][tid] = j < count ? fmaf(q.x, k.x, q.y * k.y) : 0.0f;
      }
      // Share six barriers across eight independent 32-lane FP32 trees.
      __syncthreads();
      const int lane = tid % 32;
      for (int offset = 16; offset > 0; offset >>= 1) {
        if (lane < offset) {
#pragma unroll
          for (int j = 0; j < kTokens; ++j) {
            reductions[j][tid] += reductions[j][tid + offset];
          }
        }
        __syncthreads();
      }
      if (tid < 32) {
        float dot = 0.0f;
        if (tid < count) {
          dot += reductions[tid][0];
          dot += reductions[tid][32];
        }
        const float score = tid < count ? dot * scale : -INFINITY;
        scores[tid] = score;
        if (tid < kTokens) weights[tid] = score;
      }
      __syncthreads();
      for (int offset = 16; offset > 0; offset >>= 1) {
        if (tid < offset)
          scores[tid] = fmaxf(scores[tid], scores[tid + offset]);
        __syncthreads();
      }
      const float next_max = fmaxf(maximum, scores[0]);
      __syncthreads();
      if (tid < 32) {
        const float weight =
            tid < count ? exp2f(weights[tid] - next_max) : 0.0f;
        scores[tid] = weight;
        if (tid < kTokens) weights[tid] = weight;
      }
      __syncthreads();
      for (int offset = 16; offset > 0; offset >>= 1) {
        if (tid < offset) scores[tid] += scores[tid + offset];
        __syncthreads();
      }
      if (tid == 0) alpha = exp2f(maximum - next_max);
      __syncthreads();
      sum = sum * alpha + scores[0];
      maximum = next_max;
      float value0 = 0.0f, value1 = 0.0f;
#pragma unroll
      for (int j = 0; j < kTokens; ++j) {
        const float2 v = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(&values[j][dim]));
        value0 = fmaf(weights[j], v.x, value0);
        value1 = fmaf(weights[j], v.y, value1);
      }
      acc0 = acc0 * alpha + value0;
      acc1 = acc1 * alpha + value1;
      __syncthreads();
    }
  }
  if (tid == 0) {
    params.partial_max[row] = maximum;
    params.partial_sum[row] = sum;
  }
  params.partial_acc[row * kHeadSize + dim] = acc0;
  params.partial_acc[row * kHeadSize + dim + 1] = acc1;
}

template <int kHeadSize>
__global__ void DecodeCombineKernel(DecodeQkParams params) {
  const int row = blockIdx.y * gridDim.x + blockIdx.x;
  const int rows = gridDim.x * gridDim.y;
  __shared__ float denominator;
  __shared__ float split_weights[8];
  if (threadIdx.x == 0) {
    float maximum = -INFINITY;
    for (int split = 0; split < params.splits; ++split) {
      maximum = fmaxf(maximum, params.partial_max[split * rows + row]);
    }
    float sum = 0.0f;
    for (int split = 0; split < params.splits; ++split) {
      const int index = split * rows + row;
      split_weights[split] = 0.0f;
      if (params.partial_sum[index] > 0.0f) {
        const float weight = exp2f(params.partial_max[index] - maximum);
        split_weights[split] = weight;
        sum += params.partial_sum[index] * weight;
      }
    }
    denominator = 1.0f / (sum + 1e-6f);
  }
  __syncthreads();
  auto* out = params.out + blockIdx.y * params.config.out_batch_stride +
              blockIdx.x * params.config.out_head_stride;
  for (int dim = threadIdx.x; dim < kHeadSize; dim += blockDim.x) {
    float acc = 0.0f;
    for (int split = 0; split < params.splits; ++split) {
      const int index = split * rows + row;
      acc += params.partial_acc[index * kHeadSize + dim] * split_weights[split];
    }
    out[dim] = __float2bfloat16(acc * denominator);
  }
}

}  // namespace infini::ops::flash_attn_with_kvcache_iluvatar_detail

#endif
