#pragma once
#include <cuda_runtime.h>

#include <cstdint>
namespace infini::ops::deepseek_v4_w8a8_detail {
// HCU shared dense projection switches to its assembly epilogue above 16
// tokens. Routed GEMMs always use the joint-scale epilogue.
static __device__ float ScaledDot(int32_t accumulator, float token_scale,
                                  float weight_scale, bool joint_scale) {
  if (!joint_scale)
    return __fmul_rn(__fmul_rn(static_cast<float>(accumulator), token_scale),
                     weight_scale);
  const float scale = __fmul_rn(token_scale, weight_scale);
  return __fmul_rn(static_cast<float>(accumulator), scale);
}

}  // namespace infini::ops::deepseek_v4_w8a8_detail
