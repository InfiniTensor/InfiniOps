#pragma once

#include <cuda_runtime.h>

#include "native/cuda/hygon/caster.cuh"

namespace infini::ops::hygon_dynamic_int8 {

// Keep each fixed-width row in registers while four wave64 groups reduce it.
template <typename T, int Width, int Threads = 256>
__global__ void QuantizeFixed(const T* input, int8_t* output, float* scales,
                              size_t rows) {
  constexpr int kWave = 64;
  constexpr int kValues = Width / Threads;
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  __shared__ float wave_maxima[Threads / kWave];
  for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    float values[kValues];
    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < kValues; ++i) {
      values[i] = Caster<Device::Type::kHygon>::template Cast<float>(
          input[row * Width + tid + i * Threads]);
      maximum = fmaxf(maximum, fabsf(values[i]));
    }
#if defined(__gfx936__)
    uint32_t a = __float_as_uint(maximum);
    uint32_t b = __builtin_amdgcn_mov_dpp(a, 0xb1, 0xf, 0xf, false);
    a = a > b ? a : b;
    b = __builtin_amdgcn_mov_dpp(a, 0x4e, 0xf, 0xf, false);
    a = a > b ? a : b;
    b = __builtin_amdgcn_mov_dpp(a, 0x141, 0xf, 0xf, false);
    a = a > b ? a : b;
    b = __builtin_amdgcn_mov_dpp(a, 0x140, 0xf, 0xf, false);
    a = a > b ? a : b;
    b = __shfl_xor_sync(0xffffffffffffffffull, a, 16, kWave);
    a = a > b ? a : b;
    b = __shfl_xor_sync(0xffffffffffffffffull, a, 32, kWave);
    maximum = __uint_as_float(a > b ? a : b);
#else
#pragma unroll
    for (int offset = kWave / 2; offset; offset >>= 1) {
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffffffffffull, maximum,
                                               offset, kWave));
    }
#endif
    if (lane == 0) wave_maxima[tid / kWave] = maximum;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < Threads / kWave; ++i) {
      maximum = fmaxf(maximum, wave_maxima[i]);
    }
    // Preserve the new API's direct ratio, scale floor and ties-to-even rule.
    const float absmax = fmaxf(maximum, 1.0e-10f);
    const float scale = absmax / 127.0f;
    const float inverse = 127.0f / absmax;
    if (tid == 0) scales[row] = scale;
#pragma unroll
    for (int i = 0; i < kValues; ++i) {
      const int q = __float2int_rn(values[i] * inverse);
      output[row * Width + tid + i * Threads] =
          static_cast<int8_t>(min(max(q, -127), 127));
    }
    // Blocks may process several rows when the grid is capped.
    __syncthreads();
  }
}

}  // namespace infini::ops::hygon_dynamic_int8
