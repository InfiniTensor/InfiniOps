#ifndef INFINI_OPS_ILUVATAR_GEMM_KERNEL_CUH_
#define INFINI_OPS_ILUVATAR_GEMM_KERNEL_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace infini::ops::iluvatar {

template <int kBatch, bool kHasTail>
__global__ void SmallBatchGemvKernel(const __nv_bfloat16* x,
                                     const __nv_bfloat16* w, __nv_bfloat16* y,
                                     std::size_t input, std::size_t outputs) {
  static_assert(kBatch == 1 || kBatch == 2);
  const int lane = threadIdx.x & 31;
  const std::size_t output = blockIdx.x * 4 + threadIdx.x / 32;
  const auto* w2 = reinterpret_cast<const __nv_bfloat162*>(w + output * input);
  const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
  float s0[4] = {0.f, 0.f, 0.f, 0.f};
  float s1[4] = {0.f, 0.f, 0.f, 0.f};
  for (std::size_t k = lane; k < input / 2; k += 128) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if constexpr (kHasTail) {
        if (k + j * 32 >= input / 2) {
          continue;
        }
      }
      float2 a = __bfloat1622float2(w2[k + j * 32]);
      float2 b = __bfloat1622float2(x2[k + j * 32]);
      s0[j] = fmaf(a.x, b.x, s0[j]);
      s0[j] = fmaf(a.y, b.y, s0[j]);
      if constexpr (kBatch == 2) {
        float2 c = __bfloat1622float2(x2[input / 2 + k + j * 32]);
        s1[j] = fmaf(a.x, c.x, s1[j]);
        s1[j] = fmaf(a.y, c.y, s1[j]);
      }
    }
  }
  // Keep the four-accumulator FP32 tree used in the measured specialization.
  float a = (s0[0] + s0[1]) + (s0[2] + s0[3]);
  float b = (s1[0] + s1[1]) + (s1[2] + s1[3]);
  for (int delta = 16; delta > 0; delta /= 2) {
    a += __shfl_down_sync(0xffffffffu, a, delta);
    if constexpr (kBatch == 2) {
      b += __shfl_down_sync(0xffffffffu, b, delta);
    }
  }
  if (lane == 0) {
    y[output] = __float2bfloat16(a);
    if constexpr (kBatch == 2) {
      y[outputs + output] = __float2bfloat16(b);
    }
  }
}

}  // namespace infini::ops::iluvatar

#endif
