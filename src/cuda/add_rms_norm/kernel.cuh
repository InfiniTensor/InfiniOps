#ifndef INFINI_OPS_CUDA_ADD_RMS_NORM_KERNEL_CUH_
#define INFINI_OPS_CUDA_ADD_RMS_NORM_KERNEL_CUH_

#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>

#include "cuda/caster.cuh"
#include "cuda/kernel_commons.cuh"

namespace infini::ops {

// Fused add + RMS normalization:
//   residual_out[b, h, i] = input[b, h, i] + other[b, h, i]
//   out[b, h, i]          = residual_out[b, h, i] * weight[i] * rms
// where `rms = rsqrt(mean(residual_out[b, h]^2) + epsilon)`. Each CUDA block
// handles one `(batch, head)` row so the reduce fits inside block-local
// `cub::BlockReduce`.
template <unsigned int block_size, Device::Type kDev, typename TCompute,
          typename TData, typename TWeight>
__global__ void AddRmsNormKernel(
    TData* __restrict__ out, TData* __restrict__ residual_out,
    int64_t stride_out_batch, int64_t stride_out_nhead,
    int64_t stride_residual_out_batch, int64_t stride_residual_out_nhead,
    const TData* __restrict__ input, int64_t stride_input_batch,
    int64_t stride_input_nhead, const TData* __restrict__ other,
    int64_t stride_other_batch, int64_t stride_other_nhead,
    const TWeight* __restrict__ w, size_t nhead, size_t dim, float epsilon) {
  size_t batch_idx = blockIdx.x / nhead;
  size_t head_idx = blockIdx.x % nhead;

  auto out_ptr =
      out + batch_idx * stride_out_batch + head_idx * stride_out_nhead;
  auto residual_out_ptr = residual_out +
                          batch_idx * stride_residual_out_batch +
                          head_idx * stride_residual_out_nhead;
  auto input_ptr =
      input + batch_idx * stride_input_batch + head_idx * stride_input_nhead;
  auto other_ptr =
      other + batch_idx * stride_other_batch + head_idx * stride_other_nhead;

  // Stream the add and the sum-of-squares in a single pass. `residual_out`
  // is written immediately so downstream layers can consume it without a
  // separate kernel launch.
  TCompute ss = 0;
  for (size_t i = threadIdx.x; i < dim; i += block_size) {
    TCompute sum = Caster<kDev>::template Cast<TCompute>(input_ptr[i]) +
                   Caster<kDev>::template Cast<TCompute>(other_ptr[i]);
    residual_out_ptr[i] = Caster<kDev>::template Cast<TData>(sum);
    ss += sum * sum;
  }

  using BlockReduce = cub::BlockReduce<TCompute, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  ss = BlockReduce(temp_storage).Sum(ss);

  __shared__ TCompute rms;
  if (threadIdx.x == 0) {
    rms = Caster<kDev>::template Cast<TCompute>(
        rsqrtf(ss / Caster<kDev>::template Cast<TCompute>(dim) + epsilon));
  }
  __syncthreads();

  for (size_t i = threadIdx.x; i < dim; i += block_size) {
    out_ptr[i] = Caster<kDev>::template Cast<TData>(
        Caster<kDev>::template Cast<TCompute>(residual_out_ptr[i]) *
        Caster<kDev>::template Cast<TCompute>(w[i]) * rms);
  }
}

}  // namespace infini::ops

#endif
