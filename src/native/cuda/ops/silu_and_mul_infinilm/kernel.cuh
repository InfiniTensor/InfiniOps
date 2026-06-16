#ifndef INFINI_OPS_CUDA_SILU_AND_MUL_INFINILM_KERNEL_CUH_
#define INFINI_OPS_CUDA_SILU_AND_MUL_INFINILM_KERNEL_CUH_

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "native/cuda/caster.cuh"

namespace infini::ops {

namespace {

template <Device::Type kDev, typename T>
__device__ __forceinline__ T SiluAndMulInfinilmValue(T gate, T up) {
  if constexpr (std::is_same_v<T, double>) {
    return (gate / (1.0 + exp(-gate))) * up;
  } else {
    const float g = Caster<kDev>::template Cast<float>(gate);
    const float u = Caster<kDev>::template Cast<float>(up);
    const float y = (g / (1.0f + expf(-g))) * u;
    return Caster<kDev>::template Cast<T>(y);
  }
}

}  // namespace

template <Device::Type kDev, typename T, unsigned int block_size>
__global__ void SiluAndMulInfinilmKernel(T* __restrict__ out,
                                         const T* __restrict__ input,
                                         size_t output_size,
                                         size_t hidden_size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < output_size) {
    size_t col = idx % hidden_size;
    size_t row = idx / hidden_size;
    size_t input_base = row * hidden_size * 2;
    out[idx] = SiluAndMulInfinilmValue<kDev>(
        input[input_base + col], input[input_base + hidden_size + col]);
  }
}

}  // namespace infini::ops

#endif
