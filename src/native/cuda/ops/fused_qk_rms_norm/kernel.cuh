#ifndef INFINI_OPS_CUDA_FUSED_QK_RMS_NORM_KERNEL_CUH_
#define INFINI_OPS_CUDA_FUSED_QK_RMS_NORM_KERNEL_CUH_

#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>

#include "native/cuda/caster.cuh"
#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {

// One block per (token, head). Normalizes one row of `q` (weight =
// q_weight) or `kv` (weight = kv_weight):
// y = x * rsqrt(mean(x^2) + eps) * weight[head], fp32 accumulation.
template <unsigned int kBlockSize, Device::Type kDev, typename TData,
          typename TCompute>
__global__ void FusedQkRmsNormKernel(
    const TData* __restrict__ q, const TData* __restrict__ kv,
    const TData* __restrict__ weight_q, const TData* __restrict__ weight_kv,
    TData* __restrict__ q_output, TData* __restrict__ kv_output, float eps,
    unsigned int q_heads, unsigned int kv_heads, unsigned int head_dim,
    int64_t q_token_stride, int64_t kv_token_stride,
    int64_t q_output_token_stride, int64_t kv_output_token_stride) {
  const auto token = static_cast<int64_t>(blockIdx.y);
  const auto is_q = blockIdx.z == 0;
  const auto head = blockIdx.x;

  const TData* input;
  const TData* weight;
  TData* output;
  int64_t token_stride;
  int64_t output_token_stride;
  if (is_q) {
    if (head >= q_heads) {
      return;
    }
    input = q;
    weight = weight_q;
    output = q_output;
    token_stride = q_token_stride;
    output_token_stride = q_output_token_stride;
  } else {
    if (head >= kv_heads) {
      return;
    }
    input = kv;
    weight = weight_kv;
    output = kv_output;
    token_stride = kv_token_stride;
    output_token_stride = kv_output_token_stride;
  }

  const TData* row_ptr =
      input + token * token_stride + static_cast<int64_t>(head) * head_dim;
  TData* out_ptr = output + token * output_token_stride +
                   static_cast<int64_t>(head) * head_dim;
  // q_weight/kv_weight are single [head_dim] vectors shared by all heads.
  const TData* weight_ptr = weight;

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
    out_ptr[i] = Caster<kDev>::template Cast<TData>(
        value * shared_scale *
        Caster<kDev>::template Cast<TCompute>(weight_ptr[i]));
  }
}

}  // namespace infini::ops

#endif
