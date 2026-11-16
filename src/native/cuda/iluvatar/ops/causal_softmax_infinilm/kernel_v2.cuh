#ifndef INFINI_OPS_ILUVATAR_CAUSAL_SOFTMAX_KERNEL_V2_CUH_
#define INFINI_OPS_ILUVATAR_CAUSAL_SOFTMAX_KERNEL_V2_CUH_
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cub/block/block_reduce.cuh>
namespace infini::ops::iluvatar {
using BF = __nv_bfloat16;
struct CausalFp32Max {
  __device__ float operator()(float a, float b) const { return a > b ? a : b; }
};
template <int B>
__device__ float CausalFp32BlockMax(float x) {
  using R = cub::BlockReduce<float, B>;
  __shared__ typename R::TempStorage temp;
  return R(temp).Reduce(x, CausalFp32Max());
}
template <int B>
__device__ float CausalFp32BlockSum(float x) {
  using R = cub::BlockReduce<float, B>;
  __shared__ typename R::TempStorage temp;
  return R(temp).Sum(x);
}
// Same 1024-thread FP32 max/exp/sum and BF16 store/divide order as the pinned
// SoftmaxInfinilmKernel. Causal values are selected in registers instead of
// materializing Fill/Triu/Tril/Add outputs. No mask workspace or host
// synchronization.
template <int kBlockSize>
__global__ void CausalSoftmaxFp32Kernel(BF* out, const BF* input, size_t rows,
                                        size_t q, size_t k) {
  size_t row = blockIdx.x + size_t(blockIdx.y) * gridDim.x;
  if (row >= rows) return;
  size_t base = row * k, valid = k - q + row % q + 1;
  float thread_max = -FLT_MAX;
  for (size_t i = threadIdx.x; i < k; i += kBlockSize) {
    float value = i < valid ? __bfloat162float(input[base + i]) : -INFINITY;
    thread_max = thread_max > value ? thread_max : value;
  }
  float block_max = CausalFp32BlockMax<kBlockSize>(thread_max);
  __shared__ float maximum;
  if (threadIdx.x == 0) maximum = block_max;
  __syncthreads();
  float thread_sum = 0.f;
  for (size_t i = threadIdx.x; i < k; i += kBlockSize) {
    float value = i < valid ? __bfloat162float(input[base + i]) : -INFINITY;
    float e = expf(value - maximum);
    thread_sum += e;
    out[base + i] = __float2bfloat16(e);
  }
  float block_sum = CausalFp32BlockSum<kBlockSize>(thread_sum);
  __shared__ float sum;
  if (threadIdx.x == 0) sum = block_sum;
  __syncthreads();
  for (size_t i = threadIdx.x; i < k; i += kBlockSize)
    out[base + i] = __float2bfloat16(__bfloat162float(out[base + i]) / sum);
}

}  // namespace infini::ops::iluvatar

#endif
