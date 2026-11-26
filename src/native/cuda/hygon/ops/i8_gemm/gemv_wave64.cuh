#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "native/cuda/hygon/caster.cuh"

namespace infini::ops::hygon_i8_gemm {

template <int K, typename Tdata>
__global__ void gemv_i8_hygon(Tdata *__restrict__ y, const Tdata *bias,
                              const int8_t *__restrict__ x,
                              const float *x_scale,
                              const int8_t *__restrict__ w,
                              const float *w_scale, int n, ptrdiff_t bias_col) {
  const int lane = threadIdx.x & 63;
  const int col = blockIdx.x * (blockDim.x / 64) + threadIdx.x / 64;
  if (col >= n) return;
  const int32_t *xp = reinterpret_cast<const int32_t *>(x);
  const int32_t *wp =
      reinterpret_cast<const int32_t *>(w + static_cast<int64_t>(col) * K);
  int32_t sum = 0;
#pragma unroll
  for (int step = 0; step < K / 256; ++step) {
    const int i = lane + step * 64;
    sum = __builtin_amdgcn_sdot4(__ldg(xp + i), __ldg(wp + i), sum, false);
  }
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffffffffffffull, sum, offset, 64);
  if (lane == 0) {
    const float scale = __fmul_rn(x_scale[0], w_scale[col]);
    float output = __fmul_rn(scale, static_cast<float>(sum));
    if (bias != nullptr)
      output += Caster<Device::Type::kHygon>::template Cast<float>(
          bias[col * bias_col]);
    y[col] = Caster<Device::Type::kHygon>::template Cast<Tdata>(output);
  }
}
// Aligned 16-byte loads reduce load instructions for the measured BW decode
// shape.
template <int K, typename Tdata>
__global__ void gemv_vector_hygon(Tdata *__restrict__ y, const Tdata *bias,
                                  const int8_t *__restrict__ x,
                                  const float *x_scale,
                                  const int8_t *__restrict__ w,
                                  const float *w_scale, int n,
                                  ptrdiff_t bias_col) {
  const int lane = threadIdx.x & 63;
  const int col = blockIdx.x * (blockDim.x / 64) + threadIdx.x / 64;
  if (col >= n) return;
  const int4 *xp = reinterpret_cast<const int4 *>(x);
  const int4 *wp =
      reinterpret_cast<const int4 *>(w + static_cast<int64_t>(col) * K);
  int32_t sum = 0;
#pragma unroll
  for (int step = 0; step < K / 1024; ++step) {
    const int i = lane + step * 64;
    const int4 a = __ldg(xp + i), b = __ldg(wp + i);
    sum = __builtin_amdgcn_sdot4(a.x, b.x, sum, false);
    sum = __builtin_amdgcn_sdot4(a.y, b.y, sum, false);
    sum = __builtin_amdgcn_sdot4(a.z, b.z, sum, false);
    sum = __builtin_amdgcn_sdot4(a.w, b.w, sum, false);
  }
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffffffffffffull, sum, offset, 64);
  if (lane == 0) {
    const float scale = __fmul_rn(x_scale[0], w_scale[col]);
    float output = __fmul_rn(scale, static_cast<float>(sum));
    if (bias != nullptr)
      output += Caster<Device::Type::kHygon>::template Cast<float>(
          bias[col * bias_col]);
    y[col] = Caster<Device::Type::kHygon>::template Cast<Tdata>(output);
  }
}

}  // namespace infini::ops::hygon_i8_gemm
