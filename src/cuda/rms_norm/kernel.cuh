#ifndef INFINI_OPS_CUDA_RMS_NORM_KERNEL_CUH_
#define INFINI_OPS_CUDA_RMS_NORM_KERNEL_CUH_

#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>

#include "cuda/caster.cuh"
#include "cuda/kernel_commons.cuh"

namespace infini::ops {

// Single-pass RmsNorm kernel with shared memory caching.
//
// Pass 1: Load x from global memory into shared memory, accumulate
//         sum-of-squares in registers, then block-reduce.
// Pass 2: Read x from shared memory (NOT global), apply rms * weight,
//         write y to global memory.
//
// This halves global memory traffic compared to the two-pass approach.
template <unsigned int block_size, Device::Type kDev, typename TCompute,
          typename TData, typename TWeight>
__global__ void RmsNormKernel(TData* __restrict__ y, int64_t stride_y_batch,
                              int64_t stride_y_nhead,
                              const TData* __restrict__ x,
                              int64_t stride_x_batch, int64_t stride_x_nhead,
                              const TWeight* __restrict__ w, size_t nhead,
                              size_t dim, float epsilon) {
  // Dynamic shared memory: [dim] elements of TCompute for caching x.
  extern __shared__ char smem_raw[];
  TCompute* x_cache = reinterpret_cast<TCompute*>(smem_raw);

  size_t batch_idx = blockIdx.x / nhead;
  size_t head_idx = blockIdx.x % nhead;

  auto y_ptr = y + batch_idx * stride_y_batch + head_idx * stride_y_nhead;
  auto x_ptr = x + batch_idx * stride_x_batch + head_idx * stride_x_nhead;

  // Pass 1: Load x into shared memory and compute sum-of-squares.
  TCompute ss = 0;

  for (size_t i = threadIdx.x; i < dim; i += block_size) {
    TCompute val = Caster<kDev>::template Cast<TCompute>(x_ptr[i]);
    x_cache[i] = val;
    ss += val * val;
  }

  // Block reduce sum-of-squares.
  // Place CUB temp storage after the x_cache region to avoid overlap.
  using BlockReduce = cub::BlockReduce<TCompute, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  TCompute total = BlockReduce(temp_storage).Sum(ss);

  __shared__ TCompute rms;

  if (threadIdx.x == 0) {
    rms = rsqrtf(total / static_cast<TCompute>(dim) + epsilon);
  }

  __syncthreads();

  // Pass 2: Transform using cached x from shared memory (no second
  // global read).
  for (size_t i = threadIdx.x; i < dim; i += block_size) {
    y_ptr[i] = Caster<kDev>::template Cast<TData>(
        x_cache[i] *
        Caster<kDev>::template Cast<TCompute>(w[i]) * rms);
  }
}

}  // namespace infini::ops

#endif
