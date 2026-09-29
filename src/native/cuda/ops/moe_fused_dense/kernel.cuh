#ifndef INFINI_OPS_CUDA_MOE_FUSED_DENSE_KERNEL_CUH_
#define INFINI_OPS_CUDA_MOE_FUSED_DENSE_KERNEL_CUH_

#include <cmath>

#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {

// Computes the inclusive prefix sum of `counts[0..num_experts-1]`, storing the
// exclusive prefix in `offsets[0..num_experts-1]` and the total in
// `offsets[num_experts]`. Only one thread (thread 0) is used.
__global__ void ExclusivePrefixCountsKernel(const int* counts, int* offsets,
                                            int num_experts) {
  if (threadIdx.x == 0) {
    offsets[0] = 0;
    int sum = 0;
    for (int i = 0; i < num_experts; ++i) {
      sum += counts[i];
      offsets[i + 1] = sum;
    }
  }
}

// For each aligned block of `block_size` tokens, atomically adds `block_size`
// to the count of the expert that owns the block. Rows whose expert id is out
// of range (padding rows) are skipped.
// NOTE: `num_tokens_post_padded` is a host-side scalar (not a device pointer),
// because Metax does not support device-side pointer dereference inside
// kernels.
__global__ void CountAlignedExpertsKernel(const int* expert_ids,
                                          int num_tokens_post_padded,
                                          int* counts, int num_experts,
                                          int block_size) {
  int block = blockIdx.x * blockDim.x + threadIdx.x;
  int num_blocks = (num_tokens_post_padded + block_size - 1) / block_size;
  if (block >= num_blocks) {
    return;
  }
  int expert = expert_ids[block];
  if (expert >= 0 && expert < num_experts) {
    atomicAdd(counts + expert, block_size);
  }
}

// Gathers hidden states into a packed, expert-bucketed buffer. Each output row
// `row` maps to `pair = sorted_token_ids[row]`, and its hidden state is copied
// from the source token `pair / topk`. Rows pointing past the valid pair range
// (padding) are zero-filled.
template <Device::Type kDev, typename T>
__global__ void PackHiddenAlignedKernel(const T* hidden,
                                        const int* sorted_token_ids,
                                        int* output_permutation,
                                        T* packed_hidden, int pairs, int topk,
                                        int hidden_size,
                                        int max_num_tokens_padded) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= max_num_tokens_padded) {
    return;
  }
  int pair = sorted_token_ids[row];
  if (pair >= 0 && pair < pairs) {
    if (tid == 0) {
      output_permutation[pair] = row;
    }
    int token = pair / topk;
    for (int h = tid; h < hidden_size; h += blockDim.x) {
      packed_hidden[static_cast<size_t>(row) * hidden_size + h] =
          hidden[static_cast<size_t>(token) * hidden_size + h];
    }
  } else {
    for (int h = tid; h < hidden_size; h += blockDim.x) {
      packed_hidden[static_cast<size_t>(row) * hidden_size + h] =
          Caster<kDev>::template Cast<T>(0.0f);
    }
  }
}

// SwiGLU activation: out = up * silu(gate), where silu(x) = x / (1 + exp(-x)).
template <Device::Type kDev, typename T>
__global__ void SwigluKernel(const T* gate_up, T* activated, int rows,
                             int intermediate_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * intermediate_size;
  if (idx >= total) {
    return;
  }
  int row = idx / intermediate_size;
  int col = idx - row * intermediate_size;
  const T* base = gate_up + static_cast<size_t>(row) * intermediate_size * 2;
  float gate = Caster<kDev>::template Cast<float>(base[col]);
  float up = Caster<kDev>::template Cast<float>(base[intermediate_size + col]);
  float silu = gate / (1.0f + expf(-gate));
  activated[idx] = Caster<kDev>::template Cast<T>(up * silu);
}

// Scatters the weighted expert outputs back to the original token rows. For
// each token `token`, sums over its `topk` pairs, gathering rows through
// `output_permutation` and weighting by `topk_weights`.
template <Device::Type kDev, typename T>
__global__ void ApplyShuffleMulSumKernel(
    const T* __restrict__ expert_out, T* __restrict__ out,
    const int* __restrict__ output_permutation,
    const float* __restrict__ topk_weights, int num_tokens, int topk,
    int hidden_size) {
  int token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }

  for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
    float sum = 0.0f;
    for (int k = 0; k < topk; ++k) {
      int pair = token * topk + k;
      int src_row = output_permutation[pair];
      if (src_row >= 0) {
        sum += Caster<kDev>::template Cast<float>(
                   expert_out[static_cast<size_t>(src_row) * hidden_size + h]) *
               topk_weights[pair];
      }
    }
    out[static_cast<size_t>(token) * hidden_size + h] =
        Caster<kDev>::template Cast<T>(sum);
  }
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_MOE_FUSED_DENSE_KERNEL_CUH_