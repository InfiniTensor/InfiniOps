#ifndef INFINI_OPS_CUDA_CAUSAL_SOFTMAX_KERNEL_CUH_
#define INFINI_OPS_CUDA_CAUSAL_SOFTMAX_KERNEL_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>
#include <cub/block/block_reduce.cuh>

namespace infini::ops {

namespace {

template <typename Tdata, typename Tcompute>
__device__ __forceinline__ Tdata expAndCast(Tcompute x) {
  Tcompute e = std::exp(x);
  if constexpr (std::is_same_v<Tdata, half>) {
    return __float2half(static_cast<float>(e));
  } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
    return __float2bfloat16(static_cast<float>(e));
  } else {
    return static_cast<Tdata>(e);
  }
}

struct BlockMaxOp {
  template <typename T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return (a > b) ? a : b;
  }
};

template <unsigned int BLOCK_SIZE, typename Tdata>
__device__ __forceinline__ Tdata blockMax(const Tdata* data_ptr, size_t count) {
  Tdata thread_max = count > 0 ? data_ptr[0] : Tdata{};
  for (size_t i = threadIdx.x; i < count; i += BLOCK_SIZE) {
    Tdata v = data_ptr[i];
    thread_max = (v > thread_max) ? v : thread_max;
  }
  using BlockReduce = cub::BlockReduce<Tdata, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  return BlockReduce(temp_storage).Reduce(thread_max, BlockMaxOp());
}

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ __forceinline__ Tcompute blockSum(const Tdata* data_ptr,
                                             size_t count) {
  Tcompute thread_sum = 0;
  for (size_t i = threadIdx.x; i < count; i += BLOCK_SIZE) {
    thread_sum += Tcompute(data_ptr[i]);
  }
  using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  return BlockReduce(temp_storage).Sum(thread_sum);
}

}  // namespace

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__global__ void causalSoftmaxKernel(Tdata* __restrict__ y,
                                    const Tdata* __restrict__ x,
                                    size_t batch_size, size_t seq_len,
                                    size_t total_seq_len, int64_t y_stride_b,
                                    int64_t y_stride_i, int64_t x_stride_b,
                                    int64_t x_stride_i) {
  size_t row_idx = blockIdx.x;
  size_t batch_idx = blockIdx.y;

  Tdata* y_row =
      y + batch_idx * y_stride_b + row_idx * y_stride_i;
  const Tdata* x_row =
      x + batch_idx * x_stride_b + row_idx * x_stride_i;

  size_t valid_len = total_seq_len - seq_len + row_idx + 1;

  __shared__ Tdata max_val;
  max_val = blockMax<BLOCK_SIZE, Tdata>(x_row, valid_len);
  __syncthreads();

  for (size_t col = threadIdx.x; col < total_seq_len; col += BLOCK_SIZE) {
    if (col < valid_len) {
      Tcompute diff = static_cast<Tcompute>(x_row[col]) -
                      static_cast<Tcompute>(max_val);
      y_row[col] = expAndCast<Tdata, Tcompute>(diff);
    } else {
      y_row[col] = Tdata(0);
    }
  }
  __syncthreads();

  __shared__ Tcompute sum_val;
  sum_val = blockSum<BLOCK_SIZE, Tdata, Tcompute>(y_row, total_seq_len);
  __syncthreads();

  for (size_t col = threadIdx.x; col < total_seq_len; col += BLOCK_SIZE) {
    Tcompute quot = static_cast<Tcompute>(y_row[col]) / sum_val;
    if constexpr (std::is_same_v<Tdata, half>) {
      y_row[col] = __float2half(static_cast<float>(quot));
    } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
      y_row[col] = __float2bfloat16(static_cast<float>(quot));
    } else {
      y_row[col] = static_cast<Tdata>(quot);
    }
  }
}

}  // namespace infini::ops

#endif
