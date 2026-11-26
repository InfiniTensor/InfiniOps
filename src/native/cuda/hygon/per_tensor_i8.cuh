#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "native/cuda/hygon/caster.cuh"
#include "tensor.h"

namespace infini::ops::hygon_tensor_i8 {

using Convert = Caster<Device::Type::kHygon>;
constexpr int kThreads = 256;
constexpr int kMaxRank = 8;

struct Layout {
  size_t shape[kMaxRank]{};
  ptrdiff_t strides[kMaxRank]{};
  int rank{};
  bool contiguous{};

  __device__ size_t offset(size_t index) const {
    if (contiguous) return index;
    size_t result = 0;
    for (int dim = rank - 1; dim >= 0; --dim) {
      result += (index % shape[dim]) * strides[dim];
      index /= shape[dim];
    }
    return result;
  }
};

inline bool ValidLayout(Tensor tensor, bool output) {
  if (tensor.ndim() > kMaxRank) return false;
  std::vector<std::pair<ptrdiff_t, size_t>> dimensions;
  size_t span = 1;
  for (size_t dim = 0; dim < tensor.ndim(); ++dim) {
    if (tensor.stride(dim) < 0) return false;
    if (tensor.size(dim) > 1) {
      const size_t stride = tensor.stride(dim);
      if (stride > (PTRDIFF_MAX - span) / (tensor.size(dim) - 1)) return false;
      span += stride * (tensor.size(dim) - 1);
      dimensions.emplace_back(tensor.stride(dim), tensor.size(dim));
    }
  }
  if (output && tensor.numel()) {
    std::sort(dimensions.begin(), dimensions.end());
    size_t extent = 1;
    for (auto [stride, length] : dimensions) {
      if (static_cast<size_t>(stride) < extent) return false;
      extent += (length - 1) * stride;
    }
  }
  return true;
}

inline Layout MakeLayout(Tensor tensor) {
  Layout layout;
  layout.rank = static_cast<int>(tensor.ndim());
  layout.contiguous = tensor.IsContiguous();
  for (int dim = 0; dim < layout.rank; ++dim) {
    layout.shape[dim] = tensor.size(dim);
    layout.strides[dim] = tensor.stride(dim);
  }
  return layout;
}

inline bool IsFloat(Tensor tensor) {
  return tensor.dtype() == DataType::kFloat32 ||
         tensor.dtype() == DataType::kFloat16 ||
         tensor.dtype() == DataType::kBFloat16;
}

inline bool ScalarOnDevice(Tensor scalar, Tensor tensor) {
  return scalar.dtype() == DataType::kFloat32 && scalar.numel() == 1 &&
         scalar.IsContiguous() && scalar.device() == tensor.device();
}

__device__ __forceinline__ float BlockMaximum(float value, float* scratch) {
  const int lane = threadIdx.x % 64;
  for (int offset = 32; offset; offset >>= 1) {
    value =
        fmaxf(value, __shfl_xor_sync(0xffffffffffffffffull, value, offset, 64));
  }
  if (lane == 0) scratch[threadIdx.x / 64] = value;
  __syncthreads();
  for (int wave = 0; wave < kThreads / 64; ++wave)
    value = fmaxf(value, scratch[wave]);
  return value;
}

__device__ __forceinline__ int8_t Quantized(float value, float scale) {
  const float inverse = 1.0f / scale;
  const float normalized =
      isfinite(inverse) ? value * inverse : __fdiv_rn(value, scale);
  // Saturate before integer conversion, while retaining the new ties-to-even
  // rule.
  const float bounded = fminf(fmaxf(normalized, -127.0f), 127.0f);
  return static_cast<int8_t>(__float2int_rn(bounded));
}

template <class T>
__global__ void DynamicSmall(const T* input, int8_t* output, float* scale,
                             size_t count, Layout in_layout,
                             Layout out_layout) {
  __shared__ float scratch[kThreads / 64];
  float values[16];
  float maximum = 0.0f;
  for (int i = 0; i < 16; ++i) {
    const size_t index = threadIdx.x + i * kThreads;
    values[i] =
        index < count
            ? Convert::template Cast<float>(input[in_layout.offset(index)])
            : 0.0f;
    maximum = fmaxf(maximum, fabsf(values[i]));
  }
  const float s = fmaxf(BlockMaximum(maximum, scratch), 1.0e-10f) / 127.0f;
  if (threadIdx.x == 0) scale[0] = s;
  for (int i = 0; i < 16; ++i) {
    const size_t index = threadIdx.x + i * kThreads;
    if (index < count)
      output[out_layout.offset(index)] = Quantized(values[i], s);
  }
}

template <class T>
__global__ void AbsMaximum(const T* input, float* maximum, size_t count,
                           Layout layout) {
  __shared__ float scratch[kThreads / 64];
  float value = 0.0f;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count;
       i += size_t(blockDim.x) * gridDim.x) {
    value = fmaxf(
        value, fabsf(Convert::template Cast<float>(input[layout.offset(i)])));
  }
  value = BlockMaximum(value, scratch);
  if (threadIdx.x == 0)
    atomicMax(reinterpret_cast<unsigned*>(maximum), __float_as_uint(value));
}

static __global__ void FinalizeScale(float* scale) {
  scale[0] = fmaxf(scale[0], 1.0e-10f) / 127.0f;
}

template <class T>
__global__ void Quantize(const T* input, int8_t* output, const float* scale,
                         size_t count, Layout in_layout, Layout out_layout) {
  const float s = scale[0];
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count;
       i += size_t(blockDim.x) * gridDim.x) {
    output[out_layout.offset(i)] =
        Quantized(Convert::template Cast<float>(input[in_layout.offset(i)]), s);
  }
}

template <class T>
__global__ void Dequantize(T* output, const int8_t* input, const float* scale,
                           const float* zero, size_t count, Layout out_layout,
                           Layout in_layout) {
  const float s = scale[0];
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count;
       i += size_t(blockDim.x) * gridDim.x) {
    const float q = static_cast<float>(input[in_layout.offset(i)]);
    const float value = zero ? q - zero[0] : q;
    float product;
#if defined(__gfx936__) || defined(__gfx938__)
    // Preserve signed zero before converting to a reduced-precision output.
    asm("v_mul_f32 %0, %1, %2" : "=v"(product) : "v"(value), "v"(s));
#else
    product = __fmul_rn(value, s);
#endif
    output[out_layout.offset(i)] = Convert::template Cast<T>(product);
  }
}

}  // namespace infini::ops::hygon_tensor_i8
