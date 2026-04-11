#ifndef INFINI_OPS_CUDA_ADD_RMS_NORM_KERNEL_CUH_
#define INFINI_OPS_CUDA_ADD_RMS_NORM_KERNEL_CUH_

#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>

#include "cuda/caster.cuh"
#include "cuda/kernel_commons.cuh"

namespace infini::ops {

template <unsigned int block_size, Device::Type kDev, typename TCompute,
          typename TData, typename TWeight>
__global__ void AddRmsNormKernel(
    TData* __restrict__ y_out, int64_t stride_y_out_batch,
    int64_t stride_y_out_nhead, TData* __restrict__ x_out,
    int64_t stride_x_out_batch, int64_t stride_x_out_nhead,
    const TData* __restrict__ x1, int64_t stride_x1_batch,
    int64_t stride_x1_nhead, const TData* __restrict__ x2,
    int64_t stride_x2_batch, int64_t stride_x2_nhead,
    const TWeight* __restrict__ w, size_t nhead, size_t dim, float epsilon) {
  size_t batch_idx = blockIdx.x / nhead;
  size_t head_idx = blockIdx.x % nhead;

  auto y_out_ptr =
      y_out + batch_idx * stride_y_out_batch + head_idx * stride_y_out_nhead;
  auto x_out_ptr =
      x_out + batch_idx * stride_x_out_batch + head_idx * stride_x_out_nhead;
  auto x1_ptr = x1 + batch_idx * stride_x1_batch + head_idx * stride_x1_nhead;
  auto x2_ptr = x2 + batch_idx * stride_x2_batch + head_idx * stride_x2_nhead;

  // Pass 1: Compute residual sum and accumulate sum of squares.
  TCompute ss = 0;

  for (size_t i = threadIdx.x; i < dim; i += block_size) {
    TCompute val = Caster<kDev>::template Cast<TCompute>(x1_ptr[i]) +
                   Caster<kDev>::template Cast<TCompute>(x2_ptr[i]);
    x_out_ptr[i] = Caster<kDev>::template Cast<TData>(val);
    ss += val * val;
  }

  // Block-reduce to compute the total sum of squares.
  using BlockReduce = cub::BlockReduce<TCompute, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  ss = BlockReduce(temp_storage).Sum(ss);

  __shared__ TCompute rms;

  if (threadIdx.x == 0) {
    rms = Caster<kDev>::template Cast<TCompute>(
        rsqrtf(ss / Caster<kDev>::template Cast<TCompute>(dim) + epsilon));
  }
  __syncthreads();

  // Pass 2: Write normalized output.
  for (size_t i = threadIdx.x; i < dim; i += block_size) {
    y_out_ptr[i] = Caster<kDev>::template Cast<TData>(
        Caster<kDev>::template Cast<TCompute>(x_out_ptr[i]) *
        Caster<kDev>::template Cast<TCompute>(w[i]) * rms);
  }
}

}  // namespace infini::ops

#endif
