#ifndef INFINI_OPS_CUDA_RMS_NORM_KERNEL_CUH_
#define INFINI_OPS_CUDA_RMS_NORM_KERNEL_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>

namespace infini::ops {

namespace {

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ __forceinline__ Tcompute sumSquared(const Tdata* data_ptr,
                                               size_t count) {
  Tcompute ss = 0;
  for (size_t i = threadIdx.x; i < count; i += BLOCK_SIZE) {
    ss += Tcompute(data_ptr[i]) * Tcompute(data_ptr[i]);
  }
  using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  return BlockReduce(temp_storage).Sum(ss);
}

}  // namespace

template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata,
          typename Tweight>
__global__ void rmsnormKernel(Tdata* __restrict__ y, int64_t stride_y_batch,
                              int64_t stride_y_nhead,
                              const Tdata* __restrict__ x,
                              int64_t stride_x_batch, int64_t stride_x_nhead,
                              const Tweight* __restrict__ w, size_t nhead,
                              size_t dim, float epsilon) {
  size_t batch_idx = blockIdx.x / nhead;
  size_t head_idx = blockIdx.x % nhead;

  auto y_ptr = y + batch_idx * stride_y_batch + head_idx * stride_y_nhead;
  auto x_ptr = x + batch_idx * stride_x_batch + head_idx * stride_x_nhead;
  auto w_ptr = w;

  Tcompute ss = sumSquared<BLOCK_SIZE, Tdata, Tcompute>(x_ptr, dim);

  __shared__ Tcompute rms;
  if (threadIdx.x == 0) {
    rms = Tcompute(rsqrtf(ss / Tcompute(dim) + epsilon));
  }
  __syncthreads();

  for (size_t i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
    y_ptr[i] = Tdata(Tcompute(x_ptr[i]) * Tcompute(w_ptr[i]) * rms);
  }
}

}  // namespace infini::ops

#endif
