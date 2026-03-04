#ifndef INFINI_OPS_CUDA_RMS_NORM_KERNEL_CUH_
#define INFINI_OPS_CUDA_RMS_NORM_KERNEL_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>

namespace infini::ops {

namespace {

template <unsigned int block_size, typename Data, typename Compute>
__device__ __forceinline__ Compute sumSquared(const Data* data_ptr,
                                              size_t count) {
  Compute ss = 0;
  for (size_t i = threadIdx.x; i < count; i += block_size) {
    ss += Compute(data_ptr[i]) * Compute(data_ptr[i]);
  }
  using BlockReduce = cub::BlockReduce<Compute, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  return BlockReduce(temp_storage).Sum(ss);
}

}  // namespace

template <unsigned int block_size, typename Compute, typename Data,
          typename Weight>
__global__ void rmsnormKernel(Data* __restrict__ y, int64_t stride_y_batch,
                              int64_t stride_y_nhead,
                              const Data* __restrict__ x,
                              int64_t stride_x_batch, int64_t stride_x_nhead,
                              const Weight* __restrict__ w, size_t nhead,
                              size_t dim, float epsilon) {
  size_t batch_idx = blockIdx.x / nhead;
  size_t head_idx = blockIdx.x % nhead;

  auto y_ptr = y + batch_idx * stride_y_batch + head_idx * stride_y_nhead;
  auto x_ptr = x + batch_idx * stride_x_batch + head_idx * stride_x_nhead;
  auto w_ptr = w;

  Compute ss = sumSquared<block_size, Data, Compute>(x_ptr, dim);

  __shared__ Compute rms;
  if (threadIdx.x == 0) {
    rms = Compute(rsqrtf(ss / Compute(dim) + epsilon));
  }
  __syncthreads();

  for (size_t i = threadIdx.x; i < dim; i += block_size) {
    y_ptr[i] = Data(Compute(x_ptr[i]) * Compute(w_ptr[i]) * rms);
  }
}

}  // namespace infini::ops

#endif
