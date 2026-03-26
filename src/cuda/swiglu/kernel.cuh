#ifndef INFINI_OPS_CUDA_SWIGLU_KERNEL_CUH_
#define INFINI_OPS_CUDA_SWIGLU_KERNEL_CUH_

#include <cmath>

#include "cuda/kernel_commons.h"

namespace infini::ops {

// Optimized sigmoid function with support for vectorized types.
template <typename T>
__device__ __forceinline__ T Sigmoid(const T& x) {
  if constexpr (std::is_same_v<T, half2>) {
    return h2rcp(__hadd2(make_half2(1, 1), h2exp(__hneg2(x))));
  } else if constexpr (std::is_same_v<T, half>) {
    return hrcp(
        __hadd(half(1.f), __float2half(__expf(__half2float(__hneg(x))))));
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
    return __frcp_rn(__fadd_rn(1.0f, __expf(-x)));
  } else {
    return 1.0f / (1.0f + expf(-x));
  }
}

// SwiGLU(x, gate) = Swish(x) * gate = (x * sigmoid(x)) * gate.
template <typename T, unsigned int BLOCK_SIZE>
__global__ void SwigluKernel(T* __restrict__ out, const T* __restrict__ a,
                             const T* __restrict__ b,
                             const size_t* __restrict__ out_shape,
                             const size_t* __restrict__ input_shape,
                             const size_t* __restrict__ gate_shape,
                             const ptrdiff_t* __restrict__ out_strides,
                             const ptrdiff_t* __restrict__ input_strides,
                             const ptrdiff_t* __restrict__ gate_strides,
                             size_t output_size, size_t ndim,
                             bool out_contiguous, bool input_contiguous,
                             bool gate_contiguous) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < output_size) {
    size_t out_idx, input_idx, gate_idx;

    if (out_contiguous) {
      out_idx = idx;
    } else {
      out_idx = IndexToOffset(idx, ndim, out_shape, out_strides);
    }

    if (input_contiguous) {
      input_idx = idx;
    } else {
      input_idx = IndexToOffset(idx, ndim, input_shape, input_strides);
    }

    if (gate_contiguous) {
      gate_idx = idx;
    } else {
      gate_idx = IndexToOffset(idx, ndim, gate_shape, gate_strides);
    }

    T up = a[input_idx];
    T gate = b[gate_idx];

    if constexpr (std::is_same_v<T, half2>) {
      // Vectorized `half2` computation for better performance.
      out[out_idx] = __hmul2(__hmul2(gate, Sigmoid(gate)), up);
    } else if constexpr (std::is_same_v<T, half>) {
      // Optimized `half` precision computation.
      out[out_idx] = __hmul(__hmul(gate, Sigmoid(gate)), up);
    } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
      float gate0 = __bfloat162float(__low2bfloat16(gate));
      float gate1 = __bfloat162float(__high2bfloat16(gate));
      float up0 = __bfloat162float(__low2bfloat16(up));
      float up1 = __bfloat162float(__high2bfloat16(up));
      float sig0 = __frcp_rn(__fadd_rn(1.0f, __expf(-gate0)));
      float sig1 = __frcp_rn(__fadd_rn(1.0f, __expf(-gate1)));
      out[out_idx] =
          __floats2bfloat162_rn(__fmul_rn(__fmul_rn(gate0, sig0), up0),
                                __fmul_rn(__fmul_rn(gate1, sig1), up1));
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
      float gatef = __bfloat162float(gate);
      float upf = __bfloat162float(up);
      float sigf = __frcp_rn(__fadd_rn(1.0f, __expf(-gatef)));
      out[out_idx] =
          __float2bfloat16_rn(__fmul_rn(__fmul_rn(gatef, sigf), upf));
    } else if constexpr (std::is_same_v<T, float>) {
      out[out_idx] = __fmul_rn(__fmul_rn(gate, Sigmoid(gate)), up);
    } else {
      out[out_idx] = gate * Sigmoid(gate) * up;
    }
  }
}

}  // namespace infini::ops

#endif
