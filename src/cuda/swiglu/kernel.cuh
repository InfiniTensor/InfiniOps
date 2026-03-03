#ifndef INFINI_OPS_CUDA_SWIGLU_KERNEL_CUH_
#define INFINI_OPS_CUDA_SWIGLU_KERNEL_CUH_

#include <cmath>

#include "common/cuda/kernel_commons.h"

namespace infini::ops {

// Optimized sigmoid function with support for vectorized types
template <typename T>
__device__ __forceinline__ T sigmoid(const T& x) {
  if constexpr (std::is_same_v<T, half2>) {
    return h2rcp(__hadd2(make_half2(1, 1), h2exp(__hneg2(x))));
  } else if constexpr (std::is_same_v<T, half>) {
    return hrcp(
        __hadd(__float2half(1.f), __float2half(__expf(-__half2float(x)))));
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
    return 1.0f / (1.0f + expf(-x));
  }
}

// Optimized SwiGLU kernel following InfiniCore implementation
// SwiGLU(x, gate) = Swish(x) * gate = (x * sigmoid(x)) * gate
template <typename T, unsigned int BLOCK_SIZE>
__global__ void SwigluKernel(T* out, const T* input, const T* gate,
                             const size_t* out_shape, const size_t* input_shape,
                             const size_t* gate_shape,
                             const ptrdiff_t* out_strides,
                             const ptrdiff_t* input_strides,
                             const ptrdiff_t* gate_strides, size_t output_size,
                             size_t ndim, size_t offset, bool out_contiguous,
                             bool input_contiguous, bool gate_contiguous) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

  if (idx < output_size) {
    // Compute indices for non-contiguous tensors
    size_t out_idx, input_idx, gate_idx;

    if (out_contiguous) {
      out_idx = idx;
    } else {
      out_idx = indexToOffset(idx, ndim, out_shape, out_strides);
    }

    if (input_contiguous) {
      input_idx = idx;
    } else {
      input_idx = indexToOffset(idx, ndim, input_shape, input_strides);
    }

    if (gate_contiguous) {
      gate_idx = idx;
    } else {
      gate_idx = indexToOffset(idx, ndim, gate_shape, gate_strides);
    }

    T input_val = input[input_idx];
    T gate_val = gate[gate_idx];

    // SwiGLU(input, gate) = Swish(input) * gate
    // where Swish(x) = x * sigmoid(x)
    if constexpr (std::is_same_v<T, half2>) {
      // Vectorized half2 computation for better performance
      out[out_idx] = __hmul2(__hmul2(input_val, sigmoid(input_val)), gate_val);
    } else if constexpr (std::is_same_v<T, half>) {
      // Optimized half precision computation
      out[out_idx] = __hmul(__hmul(input_val, sigmoid(input_val)), gate_val);
    } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
      cuda_bfloat162 sig = sigmoid(input_val);
      float input0 = __bfloat162float(__low2bfloat16(input_val));
      float input1 = __bfloat162float(__high2bfloat16(input_val));
      float sig0 = __bfloat162float(__low2bfloat16(sig));
      float sig1 = __bfloat162float(__high2bfloat16(sig));
      float gate0 = __bfloat162float(__low2bfloat16(gate_val));
      float gate1 = __bfloat162float(__high2bfloat16(gate_val));
      float res0 = __fmul_rn(__fmul_rn(input0, sig0), gate0);
      float res1 = __fmul_rn(__fmul_rn(input1, sig1), gate1);
      out[out_idx] = __floats2bfloat162_rn(res0, res1);
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
      cuda_bfloat16 sig = sigmoid(input_val);
      float inputf = __bfloat162float(input_val);
      float sigf = __bfloat162float(sig);
      float gatef = __bfloat162float(gate_val);
      out[out_idx] =
          __float2bfloat16_rn(__fmul_rn(__fmul_rn(inputf, sigf), gatef));
    } else if constexpr (std::is_same_v<T, float>) {
      // Single precision float with fused multiply-add
      out[out_idx] =
          __fmul_rn(__fmul_rn(input_val, sigmoid(input_val)), gate_val);
    } else {
      // Generic fallback
      out[out_idx] = input_val * sigmoid(input_val) * gate_val;
    }
  }
}

}  // namespace infini::ops

#endif
