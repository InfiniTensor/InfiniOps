#pragma once
#include <cuda_runtime.h>
#include "native/cuda/hygon/caster.cuh"
namespace infini::ops::hygon_dynamic_int8 {
constexpr unsigned kThreads = 128;

template <typename T, bool DirectRatio = false>
__global__ void Quantize(const T* input, int8_t* output, float* scales,
                         size_t rows, size_t width) {
  __shared__ float maximum[kThreads];
  const unsigned lane = threadIdx.x;
  for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const T* in = input + row * width;
    auto* out = output + row * width;
    float local = 0.0f;
    for (size_t i = lane; i < width; i += kThreads) {
      local = fmaxf(local, fabsf(Caster<Device::Type::kHygon>::template Cast<float>(in[i])));
    }
    maximum[lane] = local;
    __syncthreads();
    for (unsigned step = kThreads / 2; step; step /= 2) {
      if (lane < step) maximum[lane] = fmaxf(maximum[lane], maximum[lane + step]);
      __syncthreads();
    }
    const float absmax = fmaxf(maximum[0], 1.0e-10f);
    const float scale = absmax / 127.0f;
    // Dense W8A8 computes the quantization multiplier from the unrounded maximum.
    const float inverse = DirectRatio ? 127.0f / absmax : 1.0f / scale;
    if (lane == 0) scales[row] = scale;
    for (size_t i = lane; i < width; i += kThreads) {
      const int q = __float2int_rn(Caster<Device::Type::kHygon>::template Cast<float>(in[i]) * inverse);
      out[i] = static_cast<int8_t>(min(max(q, -127), 127));
    }
    __syncthreads();
  }
}

} // namespace infini::ops::hygon_dynamic_int8
