/*
 * Portions of the CUDA kernels in this file are adapted from SGLang:
 * /sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu
 *
 * Copyright 2025 SGLang Team. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#ifndef INFINI_OPS_CUDA_MOE_TOPK_SOFTMAX_KERNEL_CUH_
#define INFINI_OPS_CUDA_MOE_TOPK_SOFTMAX_KERNEL_CUH_

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cub/block/block_reduce.cuh>

#include "native/cuda/caster.cuh"

namespace infini::ops {

namespace {

constexpr int kWarpSize = 32;

// Vectorized memory access helper used by the power-of-two expert kernels.
template <typename T, int N, int Alignment = sizeof(T) * N>
class alignas(Alignment) AlignedArray {
  T data[N];
};

template <Device::Type kDev, typename T>
__device__ float ConvertToFloat(T x) {
  return Caster<kDev>::template Cast<float>(x);
}

struct MoeMaxReduceOp {
  __device__ __host__ float operator()(float a, float b) const {
    return a > b ? a : b;
  }
};

}  // namespace

namespace moe {

using cub_kvp = cub::KeyValuePair<int, float>;

struct MoeTopKPair {
  static const int PAIR = 2;
  static const int MAX_INDEX = 0;
  cub_kvp max;
  cub_kvp second_max;

  __device__ MoeTopKPair() {}
  __device__ MoeTopKPair(cub_kvp max, cub_kvp second_max)
      : max(max), second_max(second_max) {}
};

// Reduces a (max, second-max) KVP pair into a single pair.
struct MoeTopKPairArgMax {
  __device__ MoeTopKPairArgMax() {}
  __device__ __forceinline__ MoeTopKPair operator()(
      const MoeTopKPair& candidate1, const MoeTopKPair& candidate2) const {
    cub_kvp global_max, global_second_max;
    if (candidate1.max.value > candidate2.max.value) {
      global_max = candidate1.max;
    } else {
      global_max = candidate2.max;
    }
    if (global_max.key == candidate1.max.key) {
      global_second_max = (candidate1.second_max.value > candidate2.max.value)
                              ? candidate1.second_max
                              : candidate2.max;
    } else {
      global_second_max = (candidate2.second_max.value > candidate1.max.value)
                              ? candidate2.second_max
                              : candidate1.max;
    }
    return MoeTopKPair(global_max, global_second_max);
  }
};

}  // namespace moe

// Generic softmax over a single row of `num_cols` experts. Writes the
// (softcapped) distribution into `output`.
template <Device::Type kDev, typename T, int TPB>
__launch_bounds__(TPB) __global__
    void MoeSoftmaxKernel(const T* input, float* output, const int num_cols,
                          const float moe_softcapping) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  __shared__ float normalizing_factor;
  __shared__ float float_max;

  const int thread_row_offset = blockIdx.x * num_cols;

  float thread_data = -FLT_MAX;
  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    float val = ConvertToFloat<kDev>(input[idx]);
    if (moe_softcapping != 0.0f) {
      val = tanhf(val / moe_softcapping) * moe_softcapping;
    }
    output[idx] = val;
    thread_data = fmaxf(val, thread_data);
  }

  const float max_elem =
      BlockReduce(tmp_storage).Reduce(thread_data, MoeMaxReduceOp());
  if (threadIdx.x == 0) {
    float_max = max_elem;
  }
  __syncthreads();

  thread_data = 0.0f;
  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    thread_data += expf(output[idx] - float_max);
  }
  const float z = BlockReduce(tmp_storage).Sum(thread_data);
  if (threadIdx.x == 0) {
    normalizing_factor = 1.0f / z;
  }
  __syncthreads();

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    output[idx] = expf(output[idx] - float_max) * normalizing_factor;
  }
}

// Two experts per pass over the block. Used together with the templated
// power-of-two kernel as the fast selection path.
template <int TPB>
__launch_bounds__(TPB) __global__
    void MoeTopKFastKernel(float* inputs_after_softmax, float* output,
                           int* indices, const int num_experts, const int k,
                           const bool renormalize,
                           const float* correction_bias) {
  using namespace moe;
  using BlockReduce = cub::BlockReduce<MoeTopKPair, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  MoeTopKPair thread_pair;

  const int block_row = blockIdx.x;
  const int thread_read_offset = blockIdx.x * num_experts;
  float row_sum_for_renormalize = 0.0f;

  for (int k_idx = 0; k_idx < (k + MoeTopKPair::PAIR - 1) / MoeTopKPair::PAIR;
       ++k_idx) {
    thread_pair.max.key = 0;
    thread_pair.max.value = -1.0f;
    thread_pair.second_max.key = 0;
    thread_pair.second_max.value = -1.0f;

    cub_kvp inp_kvp;
    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
      const int idx = thread_read_offset + expert;
      inp_kvp.key = expert;
      const float prob = inputs_after_softmax[idx];
      inp_kvp.value =
          correction_bias == nullptr ? prob : prob + correction_bias[expert];
      if (inp_kvp.value > thread_pair.max.value) {
        thread_pair.second_max = thread_pair.max;
        thread_pair.max = inp_kvp;
      } else if (inp_kvp.value > thread_pair.second_max.value) {
        thread_pair.second_max = inp_kvp;
      }
    }

    MoeTopKPairArgMax reducer;
    const MoeTopKPair result_pair =
        BlockReduce(tmp_storage).Reduce(thread_pair, reducer);
    if (threadIdx.x == 0) {
#pragma unroll
      for (int i = 0; i < MoeTopKPair::PAIR; ++i) {
        if (k_idx * 2 + i >= k) {
          break;
        }
        cub_kvp result = (i == MoeTopKPair::MAX_INDEX) ? result_pair.max
                                                       : result_pair.second_max;
        int expert = result.key;
        const float prob = inputs_after_softmax[thread_read_offset + expert];
        inputs_after_softmax[thread_read_offset + expert] = -FLT_MAX;
        int idx = k * block_row + k_idx * 2 + i;
        output[idx] = prob;
        indices[idx] = expert;
        row_sum_for_renormalize += prob;
      }
    }
    __syncthreads();
  }

  if (renormalize && threadIdx.x == 0) {
    const float inv = 1.0f / row_sum_for_renormalize;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int idx = k * block_row + k_idx;
      output[idx] *= inv;
    }
  }
}

// One expert per pass over the block. Slower than MoeTopKFastKernel but
// applies to arbitrary `k`.
template <int TPB>
__launch_bounds__(TPB) __global__
    void MoeTopKKernel(float* inputs_after_softmax, float* output, int* indices,
                       const int num_experts, const int k,
                       const bool renormalize, const float* correction_bias) {
  using cub_kvp = cub::KeyValuePair<int, float>;
  using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  cub_kvp thread_kvp;
  cub::ArgMax arg_max;

  const int block_row = blockIdx.x;
  const int thread_read_offset = blockIdx.x * num_experts;
  float row_sum_for_renormalize = 0.0f;

  for (int k_idx = 0; k_idx < k; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = -1.0f;
    cub_kvp inp_kvp;
    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
      const int idx = thread_read_offset + expert;
      inp_kvp.key = expert;
      const float prob = inputs_after_softmax[idx];
      inp_kvp.value =
          correction_bias == nullptr ? prob : prob + correction_bias[expert];
      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp =
        BlockReduce(tmp_storage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      const int expert = result_kvp.key;
      const int idx = k * block_row + k_idx;
      const float prob = inputs_after_softmax[thread_read_offset + expert];
      output[idx] = prob;
      indices[idx] = expert;
      row_sum_for_renormalize += prob;
      inputs_after_softmax[thread_read_offset + expert] = -FLT_MAX;
    }
    __syncthreads();
  }

  if (renormalize && threadIdx.x == 0) {
    const float inv = 1.0f / row_sum_for_renormalize;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int idx = k * block_row + k_idx;
      output[idx] *= inv;
    }
  }
}

// Vectorized gating-softmax + top-k for power-of-two expert counts. Each row
// is processed cooperatively by `THREADS_PER_ROW` lanes, producing both the
// softmax probabilities and the (optionally biased) top-k selection.
template <Device::Type kDev, typename T, int VPT, int NUM_EXPERTS,
          int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* kWarpSize) __global__
    void TopkGatingSoftmaxKernel(const T* input, float* output,
                                 const int num_rows, int* indices, const int k,
                                 const bool renormalize,
                                 const float moe_softcapping,
                                 const float* correction_bias) {
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;
  static constexpr int ELTS_PER_WARP = kWarpSize * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;
  static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

  static_assert(VPT % ELTS_PER_LDG == 0,
                "VPT must be a multiple of elements per load");
  static_assert(kWarpSize % THREADS_PER_ROW == 0,
                "threads per row must divide warp size");
  static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW),
                "THREADS_PER_ROW must be power of 2");
  static_assert(THREADS_PER_ROW <= kWarpSize,
                "THREADS_PER_ROW can be at most warp size");
  static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0,
                "row elements must divide warp elements");

  const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
  const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
  const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  const int thread_row = warp_base_row + thread_row_in_warp;
  if (thread_row >= num_rows) {
    return;
  }

  const T* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
  const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
  const T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

  using AccessType = AlignedArray<T, ELTS_PER_LDG>;
  T row_chunk_temp[VPT];
  auto* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk_temp);
  const auto* vec_thread_read_ptr =
      reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
  for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
    row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
  }

  float row_chunk[VPT];
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = ConvertToFloat<kDev>(row_chunk_temp[ii]);
  }

  if (moe_softcapping != 0.0f) {
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      float val = row_chunk[ii];
      if (moe_softcapping != 0.0f) {
        val = tanhf(val / moe_softcapping) * moe_softcapping;
      }
      row_chunk[ii] = val;
    }
  }

  float thread_max = row_chunk[0];
#pragma unroll
  for (int ii = 1; ii < VPT; ++ii) {
    thread_max = fmaxf(thread_max, row_chunk[ii]);
  }
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, mask,
                                                   THREADS_PER_ROW));
  }

  float row_sum = 0.0f;
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = expf(row_chunk[ii] - thread_max);
    row_sum += row_chunk[ii];
  }
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    row_sum += __shfl_xor_sync(0xffffffff, row_sum, mask, THREADS_PER_ROW);
  }
  const float reciprocal_row_sum = 1.0f / row_sum;
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] *= reciprocal_row_sum;
  }

  const int start_col = first_elt_read_by_thread;
  float row_sum_for_renormalize = 0.0f;
  for (int k_idx = 0; k_idx < k; ++k_idx) {
    float max_prob = row_chunk[0];
    float max_choice = correction_bias == nullptr
                           ? max_prob
                           : max_prob + correction_bias[start_col];
    int expert = start_col;
#pragma unroll
    for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
         ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        const int expert_idx = col + ii;
        float prob = row_chunk[ldg * ELTS_PER_LDG + ii];
        float choice = correction_bias == nullptr
                           ? prob
                           : prob + correction_bias[expert_idx];
        if (choice > max_choice) {
          max_choice = choice;
          max_prob = prob;
          expert = expert_idx;
        }
      }
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other_choice =
          __shfl_xor_sync(0xffffffff, max_choice, mask, THREADS_PER_ROW);
      float other_prob =
          __shfl_xor_sync(0xffffffff, max_prob, mask, THREADS_PER_ROW);
      int other_expert =
          __shfl_xor_sync(0xffffffff, expert, mask, THREADS_PER_ROW);
      if (other_choice > max_choice ||
          (other_choice == max_choice && other_expert < expert)) {
        max_choice = other_choice;
        max_prob = other_prob;
        expert = other_expert;
      }
    }

    if (thread_group_idx == 0) {
      const int idx = k * thread_row + k_idx;
      output[idx] = max_prob;
      indices[idx] = expert;
      row_sum_for_renormalize += max_prob;
    }

    if (k_idx + 1 < k) {
      const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
      const int thread_to_clear_in_group =
          (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
      if (thread_group_idx == thread_to_clear_in_group) {
        const int offset_for_expert = expert % ELTS_PER_LDG;
        row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] =
            -FLT_MAX;
      }
    }
  }

  if (renormalize && thread_group_idx == 0) {
    const float inv = 1.0f / row_sum_for_renormalize;
#pragma unroll
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int idx = k * thread_row + k_idx;
      output[idx] *= inv;
    }
  }
}

namespace detail {

template <typename T, int EXPERTS, int BYTES_PER_LDG>
struct MoeTopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static_assert(EXPERTS / (ELTS_PER_LDG * kWarpSize) == 0 ||
                    EXPERTS % (ELTS_PER_LDG * kWarpSize) == 0,
                "");
  static constexpr int VECS_PER_THREAD = (EXPERTS /
                                          (ELTS_PER_LDG * kWarpSize)) > 1
                                             ? (EXPERTS /
                                                (ELTS_PER_LDG * kWarpSize))
                                             : 1;
  static constexpr int VPT = VECS_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static constexpr int ROWS_PER_WARP = kWarpSize / THREADS_PER_ROW;
};

}  // namespace detail

namespace {

// Launches the vectorized power-of-two kernel for a template-computed
// configuration. `StreamT` is the backend-specific stream type.
template <Device::Type kDev, typename T, int EXPERTS, int WARPS_PER_TB,
          typename StreamT>
void LaunchTopkGatingSoftmaxHelper(const T* input, float* output, int* indices,
                                   const int num_rows, const int k,
                                   const bool renormalize,
                                   const float moe_softcapping,
                                   const float* correction_bias,
                                   StreamT stream) {
  static constexpr int MAX_BYTES_PER_LDG = 16;
  static constexpr int BYTES_PER_LDG =
      (MAX_BYTES_PER_LDG < static_cast<int>(sizeof(T)) * EXPERTS)
          ? MAX_BYTES_PER_LDG
          : static_cast<int>(sizeof(T)) * EXPERTS;
  using Constants = detail::MoeTopkConstants<T, EXPERTS, BYTES_PER_LDG>;
  static constexpr int VPT = Constants::VPT;

  if constexpr (EXPERTS > kWarpSize * VPT) {
    return;
  } else {
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;
    dim3 block_dim(kWarpSize, WARPS_PER_TB);
    TopkGatingSoftmaxKernel<kDev, T, VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG>
        <<<num_blocks, block_dim, 0, stream>>>(input, output, num_rows, indices,
                                               k, renormalize, moe_softcapping,
                                               correction_bias);
  }
}

template <Device::Type kDev, typename T, int TPB, typename StreamT>
void LaunchMoeTopkGeneric(const T* gating_output, float* topk_weights,
                          int* topk_indices, float* softmax_workspace,
                          const int num_tokens, const int num_experts,
                          const int topk, const bool renormalize,
                          const float moe_softcapping,
                          const float* correction_bias, StreamT stream) {
  assert(softmax_workspace != nullptr &&
         "moe_topk_softmax generic path requires a workspace");
  MoeSoftmaxKernel<kDev, T, TPB><<<num_tokens, TPB, 0, stream>>>(
      gating_output, softmax_workspace, num_experts, moe_softcapping);
  if (topk == 1) {
    MoeTopKKernel<TPB><<<num_tokens, TPB, 0, stream>>>(
        softmax_workspace, topk_weights, topk_indices, num_experts, topk,
        renormalize, correction_bias);
  } else {
    MoeTopKFastKernel<TPB><<<num_tokens, TPB, 0, stream>>>(
        softmax_workspace, topk_weights, topk_indices, num_experts, topk,
        renormalize, correction_bias);
  }
}

}  // namespace

// Reports whether the generic fallback path requires a device workspace buffer
// sized `num_tokens * num_experts * sizeof(float)`. Zero for the power-of-two
// fast path.
inline bool MoeTopkSoftmaxNeedsWorkspace(std::size_t num_experts) {
  const bool is_pow_2 =
      num_experts != 0 && ((num_experts & (num_experts - 1)) == 0);
  return !is_pow_2 || num_experts > 512;
}

// Host entry point dispatching to either the power-of-two vectorized kernel
// or the generic softmax + top-k fallback. `workspace` is only required when
// `MoeTopkSoftmaxNeedsWorkspace(num_experts)` is true.
template <Device::Type kDev, typename T, typename StreamT>
void LaunchMoeTopkSoftmax(const T* gating_output, float* topk_weights,
                          int* topk_indices, float* workspace,
                          const int num_tokens, const int num_experts,
                          const int topk, const bool renormalize,
                          const float moe_softcapping,
                          const float* correction_bias, StreamT stream) {
  static constexpr int WARPS_PER_TB = 4;
  if (!MoeTopkSoftmaxNeedsWorkspace(num_experts)) {
    switch (num_experts) {
      case 1:
        LaunchTopkGatingSoftmaxHelper<kDev, T, 1, WARPS_PER_TB>(
            gating_output, topk_weights, topk_indices, num_tokens, topk,
            renormalize, moe_softcapping, correction_bias, stream);
        break;
      case 2:
        LaunchTopkGatingSoftmaxHelper<kDev, T, 2, WARPS_PER_TB>(
            gating_output, topk_weights, topk_indices, num_tokens, topk,
            renormalize, moe_softcapping, correction_bias, stream);
        break;
      case 4:
        LaunchTopkGatingSoftmaxHelper<kDev, T, 4, WARPS_PER_TB>(
            gating_output, topk_weights, topk_indices, num_tokens, topk,
            renormalize, moe_softcapping, correction_bias, stream);
        break;
      case 8:
        LaunchTopkGatingSoftmaxHelper<kDev, T, 8, WARPS_PER_TB>(
            gating_output, topk_weights, topk_indices, num_tokens, topk,
            renormalize, moe_softcapping, correction_bias, stream);
        break;
      case 16:
        LaunchTopkGatingSoftmaxHelper<kDev, T, 16, WARPS_PER_TB>(
            gating_output, topk_weights, topk_indices, num_tokens, topk,
            renormalize, moe_softcapping, correction_bias, stream);
        break;
      case 32:
        LaunchTopkGatingSoftmaxHelper<kDev, T, 32, WARPS_PER_TB>(
            gating_output, topk_weights, topk_indices, num_tokens, topk,
            renormalize, moe_softcapping, correction_bias, stream);
        break;
      case 64:
        LaunchTopkGatingSoftmaxHelper<kDev, T, 64, WARPS_PER_TB>(
            gating_output, topk_weights, topk_indices, num_tokens, topk,
            renormalize, moe_softcapping, correction_bias, stream);
        break;
      case 128:
        LaunchTopkGatingSoftmaxHelper<kDev, T, 128, WARPS_PER_TB>(
            gating_output, topk_weights, topk_indices, num_tokens, topk,
            renormalize, moe_softcapping, correction_bias, stream);
        break;
      case 256:
        LaunchTopkGatingSoftmaxHelper<kDev, T, 256, WARPS_PER_TB>(
            gating_output, topk_weights, topk_indices, num_tokens, topk,
            renormalize, moe_softcapping, correction_bias, stream);
        break;
      case 512:
        LaunchTopkGatingSoftmaxHelper<kDev, T, 512, WARPS_PER_TB>(
            gating_output, topk_weights, topk_indices, num_tokens, topk,
            renormalize, moe_softcapping, correction_bias, stream);
        break;
      default:
        LaunchMoeTopkGeneric<kDev, T, 256>(
            gating_output, topk_weights, topk_indices, workspace, num_tokens,
            num_experts, topk, renormalize, moe_softcapping, correction_bias,
            stream);
        break;
    }
  } else {
    LaunchMoeTopkGeneric<kDev, T, 256>(
        gating_output, topk_weights, topk_indices, workspace, num_tokens,
        num_experts, topk, renormalize, moe_softcapping, correction_bias,
        stream);
  }
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_MOE_TOPK_SOFTMAX_KERNEL_CUH_