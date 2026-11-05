#ifndef INFINI_OPS_ILUVATAR_GEMM_KERNEL_CUH_
#define INFINI_OPS_ILUVATAR_GEMM_KERNEL_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace infini::ops::iluvatar {

template <int kBatch, int kInput, int kOutput>
__global__ void SmallBatchGemvKernel(const __nv_bfloat16* x,
                                     const __nv_bfloat16* w, __nv_bfloat16* y) {
  static_assert(kBatch == 1 || kBatch == 2);
  static_assert((kInput == 4096 && kOutput == 32768) ||
                (kInput == 16384 && kOutput == 4096));
  const int lane = threadIdx.x & 31;
  const int output = blockIdx.x * 4 + threadIdx.x / 32;
  const auto* w2 = reinterpret_cast<const __nv_bfloat162*>(w + output * kInput);
  const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
  float s0[4] = {0.f, 0.f, 0.f, 0.f};
  float s1[4] = {0.f, 0.f, 0.f, 0.f};
  for (int k = lane; k < kInput / 2; k += 128) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 a = __bfloat1622float2(w2[k + j * 32]);
      float2 b = __bfloat1622float2(x2[k + j * 32]);
      s0[j] = fmaf(a.x, b.x, s0[j]);
      s0[j] = fmaf(a.y, b.y, s0[j]);
      if constexpr (kBatch == 2) {
        float2 c = __bfloat1622float2(x2[kInput / 2 + k + j * 32]);
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
      y[kOutput + output] = __float2bfloat16(b);
    }
  }
}

}  // namespace infini::ops::iluvatar

#endif
