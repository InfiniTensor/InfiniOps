/*
 * Portions of the CUDA kernels in this file are adapted from SGLang:
 * /sgl-kernel/csrc/moe/moe_align_kernel.cu
 *
 * Copyright 2025 SGLang Team. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#ifndef INFINI_OPS_CUDA_MOE_ALIGN_KERNEL_CUH_
#define INFINI_OPS_CUDA_MOE_ALIGN_KERNEL_CUH_

#include <cstddef>
#include <cstdint>

#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {

namespace detail {

constexpr int kMoeAlignVecSize = 4;
using MoeAlignVec = int4;

constexpr std::size_t MoeAlignNextPow2(std::size_t value) {
  std::size_t result = 1;
  while (result < value) {
    result <<= 1;
  }
  return result;
}

template <typename T>
constexpr T MoeAlignCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

}  // namespace detail

template <typename scalar_t>
__global__ void MoeAlignCountAndSortExpertTokensKernel(
    const int32_t* __restrict__ topk_ids,
    const int32_t* __restrict__ expert_map,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    size_t numel) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (expert_map != nullptr) {
      expert_id = expert_id >= 0 ? expert_map[expert_id] : -1;
      if (expert_id < 0) {
        continue;
      }
    }
    expert_id += 1;
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = static_cast<int32_t>(i);
  }
}

template <typename scalar_t>
__global__ void MoeAlignBlockSizeKernel(
    const int32_t* __restrict__ topk_ids,
    const int32_t* __restrict__ expert_map,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t block_size, size_t numel, int32_t* __restrict__ cumsum,
    bool pad_sorted_token_ids, const int32_t scan_size,
    int32_t max_num_tokens_padded) {
  if (blockIdx.x == 1) {
    if (pad_sorted_token_ids) {
      using detail::kMoeAlignVecSize;
      using detail::MoeAlignVec;
      MoeAlignVec fill_vec;
      fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w =
          static_cast<int32_t>(numel);
      int32_t total_vecs =
          (max_num_tokens_padded + kMoeAlignVecSize - 1) / kMoeAlignVecSize;
      MoeAlignVec* out_ptr = reinterpret_cast<MoeAlignVec*>(sorted_token_ids);
      for (int32_t i = threadIdx.x; i < total_vecs; i += blockDim.x) {
        out_ptr[i] = fill_vec;
      }
    }
    return;
  }

  extern __shared__ int32_t smem[];
  int32_t* shared_counts = smem;
  int32_t* prefix = shared_counts + num_experts;
  int32_t* scan_buf = prefix + num_experts + 1;
  __shared__ int32_t s_total_tokens_post_pad;

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  if (tid < static_cast<size_t>(num_experts)) {
    shared_counts[tid] = 0;
  }

  __syncthreads();

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (expert_map != nullptr) {
      expert_id = expert_id >= 0 ? expert_map[expert_id] : -1;
      if (expert_id < 0) {
        continue;
      }
    }
    expert_id += 1;
    atomicAdd(&shared_counts[expert_id], 1);
  }

  __syncthreads();

  // scan_buf[i] = padded token count for expert slot i, zero elsewhere.
  // NOTE: use a shared-memory (Hillis-Steele) scan instead of warp shuffles:
  // on Metax hardware warpSize is 64 and __shfl_up_sync with a 32-bit mask
  // does not propagate values across lane 32, silently corrupting the scan.
  int32_t padded_count = 0;
  if (tid < static_cast<size_t>(num_experts)) {
    int32_t count = shared_counts[tid];
    padded_count = (count + block_size - 1) / block_size * block_size;
    scan_buf[tid] = padded_count;
  }

  // zero the tail [num_experts, scan_size) so the scan covers slots only.
  if (tid >= static_cast<size_t>(num_experts) &&
      tid < static_cast<size_t>(scan_size)) {
    scan_buf[tid] = 0;
  }
  __syncthreads();

  // block-wide inclusive scan (Hillis-Steele over shared memory;
  // correct for any warp size, unlike a shuffle-based scan).
  for (int32_t off = 1; off < scan_size; off <<= 1) {
    int32_t self = 0, add = 0;
    if (tid < static_cast<size_t>(scan_size)) {
      self = scan_buf[tid];
      if (tid >= static_cast<size_t>(off)) {
        add = scan_buf[tid - off];
      }
    }
    __syncthreads();
    if (tid < static_cast<size_t>(scan_size)) {
      scan_buf[tid] = self + add;
    }
    __syncthreads();
  }

  // exclusive prefix for slot i = inclusive_prefix[i] - padded_count[i]
  if (tid < static_cast<size_t>(num_experts)) {
    prefix[tid] = scan_buf[tid] - padded_count;
  }
  if (tid == 0) {
    prefix[num_experts] = scan_buf[num_experts - 1];
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();

  if (tid <= static_cast<size_t>(num_experts)) {
    cumsum[tid] = prefix[tid];
  }
  __syncthreads();

  const int32_t num_blocks = s_total_tokens_post_pad / block_size;
  for (int32_t i = tid; i < num_blocks; i += stride) {
    int32_t block_start = i * block_size;
    int left = 0;
    int right = num_experts;
    while (left < right) {
      int mid = (left + right) >> 1;
      if (prefix[mid] <= block_start) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    expert_ids[i] = left - 2;
  }
}

template <typename scalar_t, int32_t fill_threads>
__global__ void MoeAlignBlockSizeSmallBatchExpertKernel(
    const int32_t* __restrict__ topk_ids,
    const int32_t* __restrict__ expert_map,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t block_size, size_t numel, bool pad_sorted_token_ids,
    int32_t max_num_tokens_padded) {
  if (threadIdx.x < fill_threads) {
    if (pad_sorted_token_ids) {
      for (int32_t it = threadIdx.x; it < max_num_tokens_padded;
           it += fill_threads) {
        sorted_token_ids[it] = static_cast<int32_t>(numel);
      }
    }
    __syncthreads();
    __syncthreads();
    __syncthreads();
    return;
  }

  const size_t tid = threadIdx.x - fill_threads;
  const size_t stride = blockDim.x - fill_threads;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts =
      reinterpret_cast<int32_t*>(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(tid + 1) * num_experts + i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (expert_map != nullptr) {
      expert_id = expert_id >= 0 ? expert_map[expert_id] : -1;
      if (expert_id < 0) {
        continue;
      }
    }
    expert_id += 1;
    ++tokens_cnts[(tid + 1) * num_experts + expert_id];
  }

  __syncthreads();

  if (tid < static_cast<size_t>(num_experts)) {
    tokens_cnts[tid] = 0;
    for (size_t i = 1; i <= stride; ++i) {
      tokens_cnts[i * num_experts + tid] +=
          tokens_cnts[(i - 1) * num_experts + tid];
    }
  }

  __syncthreads();

  if (tid == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] +
                  detail::MoeAlignCeilDiv(
                      tokens_cnts[stride * num_experts + i - 1], block_size) *
                      block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  if (tid < static_cast<size_t>(num_experts)) {
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[i / block_size] = tid - 1;
    }
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (expert_map != nullptr) {
      expert_id = expert_id >= 0 ? expert_map[expert_id] : -1;
      if (expert_id < 0) {
        continue;
      }
    }
    expert_id += 1;
    int32_t rank_post_pad =
        tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = static_cast<int32_t>(i);
    ++tokens_cnts[tid * num_experts + expert_id];
  }
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_MOE_ALIGN_KERNEL_CUH_