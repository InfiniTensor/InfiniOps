#ifndef INFINI_OPS_MOORE_SWIGLU_MOORE_KERNEL_H_
#define INFINI_OPS_MOORE_SWIGLU_MOORE_KERNEL_H_

#include <cmath>
#include <type_traits>

#include <musa_bf16.h>
#include <musa_fp16.h>

namespace infini::ops::swiglu::moore {

using cuda_bfloat16 = mt_bfloat16;
using cuda_bfloat162 = mt_bfloat162;

struct SwiGLUOp {
 private:
  template <typename T>
  __device__ __forceinline__ T sigmoid(const T& x) const {
    if constexpr (std::is_same_v<T, half2>) {
      return h2rcp(__hadd2(make_half2(1, 1), h2exp(__hneg2(x))));
    } else if constexpr (std::is_same_v<T, half>) {
      float xf = __half2float(x);
      float sigf = 1.0f / (1.0f + std::exp(-xf));
      return __float2half(sigf);
    } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
      float x0 = __bfloat162float(__low2bfloat16(x));
      float x1 = __bfloat162float(__high2bfloat16(x));
      float sig0 = __frcp_rn(__fadd_rn(1.0f, __expf(-x0)));
      float sig1 = __frcp_rn(__fadd_rn(1.0f, __expf(-x1)));
      return __floats2bfloat162_rn(sig0, sig1);
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
      float xf = __bfloat162float(x);
      return __float2bfloat16_rn(__frcp_rn(__fadd_rn(1.0f, __expf(-xf))));
    } else if constexpr (std::is_same_v<T, float>) {
      return __frcp_rn(__fadd_rn(1, __expf(-x)));
    } else {
      return static_cast<T>(1 / (1 + std::exp(-x)));
    }
  }

 public:
  template <typename T>
  __device__ __forceinline__ T operator()(const T& up, const T& gate) const {
    if constexpr (std::is_same_v<T, half2>) {
      return __hmul2(__hmul2(gate, sigmoid(gate)), up);
    } else if constexpr (std::is_same_v<T, half>) {
      return __hmul(__hmul(gate, sigmoid(gate)), up);
    } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
      cuda_bfloat162 sig = sigmoid(gate);
      float gate0 = __low2float(gate);
      float gate1 = __high2float(gate);
      float sig0 = __low2float(sig);
      float sig1 = __high2float(sig);
      float up0 = __low2float(up);
      float up1 = __high2float(up);

      float res0 = __fmul_rn(__fmul_rn(gate0, sig0), up0);
      float res1 = __fmul_rn(__fmul_rn(gate1, sig1), up1);
      return __floats2bfloat162_rn(res0, res1);
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
      cuda_bfloat16 sig = sigmoid(gate);
      float gatef = __bfloat162float(gate);
      float sigf = __bfloat162float(sig);
      float upf = __bfloat162float(up);
      return __float2bfloat16_rn(__fmul_rn(__fmul_rn(gatef, sigf), upf));
    } else if constexpr (std::is_same_v<T, float>) {
      return __fmul_rn(__fmul_rn(gate, sigmoid(gate)), up);
    } else {
      return gate * sigmoid(gate) * up;
    }
  }
};

}  // namespace infini::ops::swiglu::moore

#endif  // INFINI_OPS_MOORE_SWIGLU_MOORE_KERNEL_H_
