#ifndef INFINI_OPS_CUDA_SWIGLU_KERNEL_CUH_
#define INFINI_OPS_CUDA_SWIGLU_KERNEL_CUH_

#include <cmath>

#include "cuda/kernel_commons.cuh"

namespace infini::ops {

// Optimized sigmoid function with support for FP16 and BF16 types.
// TODO: The unified FP16/BF16 branch uses `Caster` and scalar float
// arithmetic instead of native vectorized intrinsics (e.g. `h2rcp`,
// `__hmul2`). Profile and restore specialized paths if needed.
template <Device::Type kDev, typename T>
__device__ __forceinline__ T Sigmoid(const T& x) {
  if constexpr (IsFP16<kDev, T> || IsBFloat16<kDev, T>) {
    float xf = Caster<kDev>::template Cast<float>(x);
    return Caster<kDev>::template Cast<T>(
        __frcp_rn(__fadd_rn(1.0f, __expf(-xf))));
  } else if constexpr (std::is_same_v<T, float>) {
    return __frcp_rn(__fadd_rn(1.0f, __expf(-x)));
  } else {
    return 1.0f / (1.0f + expf(-x));
  }
}

// Device-side SwiGLU functor for BinaryElementwiseBrick.
// SwiGLU(input, gate) = input * gate * sigmoid(gate).
template <Device::Type kDev>
struct SwigluOp {
  template <typename T>
  __device__ __forceinline__ T operator()(const T& input,
                                          const T& gate) const {
    if constexpr (IsFP16<kDev, T> || IsBFloat16<kDev, T>) {
      float gf = Caster<kDev>::template Cast<float>(gate);
      float uf = Caster<kDev>::template Cast<float>(input);
      float sf = __frcp_rn(__fadd_rn(1.0f, __expf(-gf)));
      return Caster<kDev>::template Cast<T>(
          __fmul_rn(__fmul_rn(gf, sf), uf));
    } else if constexpr (std::is_same_v<T, float>) {
      return __fmul_rn(__fmul_rn(gate, Sigmoid<kDev>(gate)), input);
    } else {
      return gate * Sigmoid<kDev>(gate) * input;
    }
  }
};

// SwiGLU(x, gate) = Swish(x) * gate = (x * sigmoid(x)) * gate.
template <Device::Type kDev, typename T, unsigned int BLOCK_SIZE>
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

    if constexpr (IsFP16<kDev, T> || IsBFloat16<kDev, T>) {
      float gatef = Caster<kDev>::template Cast<float>(gate);
      float upf = Caster<kDev>::template Cast<float>(up);
      float sigf = __frcp_rn(__fadd_rn(1.0f, __expf(-gatef)));
      out[out_idx] = Caster<kDev>::template Cast<T>(
          __fmul_rn(__fmul_rn(gatef, sigf), upf));
    } else if constexpr (std::is_same_v<T, float>) {
      out[out_idx] = __fmul_rn(__fmul_rn(gate, Sigmoid<kDev>(gate)), up);
    } else {
      out[out_idx] = gate * Sigmoid<kDev>(gate) * up;
    }
  }
}

}  // namespace infini::ops

#endif
