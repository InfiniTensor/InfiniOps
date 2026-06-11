#ifndef INFINI_OPS_CUDA_RELU_INFINILM_KERNEL_CUH_
#define INFINI_OPS_CUDA_RELU_INFINILM_KERNEL_CUH_

#include <cstddef>

#include "native/cuda/caster.cuh"
#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {

namespace {

template <Device::Type kDev, typename T>
__device__ __forceinline__ T ReluInfinilmValue(T x) {
  const float v = Caster<kDev>::template Cast<float>(x);
  return Caster<kDev>::template Cast<T>(v > 0.0f ? v : 0.0f);
}

template <Device::Type kDev>
__device__ __forceinline__ double ReluInfinilmValue(double x) {
  return x > 0.0 ? x : 0.0;
}

}  // namespace

template <Device::Type kDev, typename T, unsigned int block_size>
__global__ void ReluInfinilmKernel(T* __restrict__ out,
                                   const T* __restrict__ input,
                                   const size_t* __restrict__ out_shape,
                                   const size_t* __restrict__ input_shape,
                                   const ptrdiff_t* __restrict__ out_strides,
                                   const ptrdiff_t* __restrict__ input_strides,
                                   size_t output_size, size_t ndim,
                                   bool out_contiguous, bool input_contiguous) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < output_size) {
    size_t out_idx =
        out_contiguous ? idx : IndexToOffset(idx, ndim, out_shape, out_strides);
    size_t input_idx =
        input_contiguous ? idx
                         : IndexToOffset(idx, ndim, input_shape, input_strides);
    out[out_idx] = ReluInfinilmValue<kDev>(input[input_idx]);
  }
}

}  // namespace infini::ops

#endif
