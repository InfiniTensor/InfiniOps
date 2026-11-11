#ifndef INFINI_OPS_CUDA_FUSED_QK_NORM_KERNEL_CUH_
#define INFINI_OPS_CUDA_FUSED_QK_NORM_KERNEL_CUH_

#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>

#include "native/cuda/caster.cuh"
#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {

// One block per (token, head). Normalizes `head_dim` contiguous bf16
// elements in place: y = x * rsqrt(mean(x^2) + eps) * weight[head].
// `is_norm_head == false` rows (V heads) are skipped entirely.
template <unsigned int kBlockSize, Device::Type kDev, typename TData,
          typename TCompute>
__global__ void FusedQkNormKernel(TData* __restrict__ qkv, int64_t head_stride,
                                  int64_t token_stride,
                                  const TData* __restrict__ q_weight,
                                  const TData* __restrict__ k_weight,
                                  unsigned int num_heads_q,
                                  unsigned int num_heads_k,
                                  unsigned int head_dim, float eps) {
  const auto token = static_cast<int64_t>(blockIdx.y);
  const auto head = blockIdx.x;
  // Heads are laid out as [Q (num_heads_q), K (num_heads_k), V (rest)];
  // only Q and K heads are normalized.
  const bool is_norm_head = head < num_heads_q + num_heads_k;
  if (!is_norm_head) {
    return;
  }
  // q_weight/k_weight are single [head_dim] vectors shared by all heads
  // of their group.
  const TData* weight_ptr = head < num_heads_q ? q_weight : k_weight;
  TData* row_ptr =
      qkv + token * token_stride + static_cast<int64_t>(head) * head_stride;

  using BlockReduce = cub::BlockReduce<TCompute, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ TCompute shared_scale;

  TCompute ss = 0;
  for (unsigned int i = threadIdx.x; i < head_dim; i += kBlockSize) {
    const TCompute value = Caster<kDev>::template Cast<TCompute>(row_ptr[i]);
    ss += value * value;
  }
  const TCompute total = BlockReduce(temp_storage).Sum(ss);
  if (threadIdx.x == 0) {
    shared_scale = Caster<kDev>::template Cast<TCompute>(
        rsqrtf(static_cast<float>(total) / static_cast<float>(head_dim) + eps));
  }
  __syncthreads();

  for (unsigned int i = threadIdx.x; i < head_dim; i += kBlockSize) {
    const TCompute value = Caster<kDev>::template Cast<TCompute>(row_ptr[i]);
    row_ptr[i] = Caster<kDev>::template Cast<TData>(
        value * shared_scale *
        Caster<kDev>::template Cast<TCompute>(weight_ptr[i]));
  }
}

}  // namespace infini::ops

#endif
