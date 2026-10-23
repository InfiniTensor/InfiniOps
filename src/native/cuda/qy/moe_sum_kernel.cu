#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "native/cuda/qy/data_type_.h"
#include "native/cuda/qy/moe_sum_kernel.h"

namespace infini::ops::qy {
namespace {

template <typename T>
__device__ __forceinline__ float toFloat(T v);

template <>
__device__ __forceinline__ float toFloat<float>(float v) {
  return v;
}

template <>
__device__ __forceinline__ float toFloat<half>(half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float toFloat<cuda_bfloat16>(cuda_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float v);

template <>
__device__ __forceinline__ float fromFloat<float>(float v) {
  return v;
}

template <>
__device__ __forceinline__ half fromFloat<half>(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ cuda_bfloat16 fromFloat<cuda_bfloat16>(float v) {
  return __float2bfloat16(v);
}

// Runtime-topk loop: one block per token, threads stride over hidden.
// Supports arbitrary topk (MiniCPM5 may use topk > 4).
template <typename T>
__global__ void moeSumKernel(T *__restrict__ output,
                             const T *__restrict__ input, size_t topk,
                             size_t hidden) {
  const size_t token_idx = blockIdx.x;
  const T *token_in = input + token_idx * topk * hidden;
  T *token_out = output + token_idx * hidden;
  for (size_t idx = threadIdx.x; idx < hidden; idx += blockDim.x) {
    float acc = 0.0f;
    for (size_t k = 0; k < topk; ++k) {
      acc += toFloat(token_in[k * hidden + idx]);
    }
    token_out[idx] = fromFloat<T>(acc);
  }
}

// Flattened grid-stride fallback when tokens exceed modest grid caps.
template <typename T>
__global__ void moeSumFlatKernel(T *__restrict__ output,
                                 const T *__restrict__ input, size_t tokens,
                                 size_t topk, size_t hidden, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (; idx < n; idx += stride) {
    size_t token = idx / hidden;
    size_t h = idx - token * hidden;
    const T *token_in = input + token * topk * hidden;
    float acc = 0.0f;
    for (size_t k = 0; k < topk; ++k) {
      acc += toFloat(token_in[k * hidden + h]);
    }
    output[idx] = fromFloat<T>(acc);
  }
}

template <typename T>
void launchMoeSum(void *output, const void *input, size_t tokens, size_t topk,
                  size_t hidden, void *stream) {
  if (tokens == 0 || hidden == 0) {
    return;
  }
  if (topk == 0) {
    throw std::runtime_error("QY moe_sum_launch: topk must be > 0");
  }

  constexpr int block = 128;
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  (void)cudaGetLastError();

  // Prefer one-block-per-token when token count is modest (Denglin grid cap).
  constexpr int kMaxTokenGrid = 128;
  if (tokens <= static_cast<size_t>(kMaxTokenGrid)) {
    int threads =
        static_cast<int>(std::min(hidden, static_cast<size_t>(block)));
    moeSumKernel<T><<<static_cast<int>(tokens), threads, 0, cuda_stream>>>(
        reinterpret_cast<T *>(output), reinterpret_cast<const T *>(input), topk,
        hidden);
  } else {
    size_t n = tokens * hidden;
    int grid = static_cast<int>((n + block - 1) / block);
    grid = std::min(grid, 128);
    moeSumFlatKernel<T><<<grid, block, 0, cuda_stream>>>(
        reinterpret_cast<T *>(output), reinterpret_cast<const T *>(input),
        tokens, topk, hidden, n);
  }

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("QY moe_sum_launch failed: ") +
                             cudaGetErrorString(err) +
                             " (tokens=" + std::to_string(tokens) +
                             ", topk=" + std::to_string(topk) +
                             ", hidden=" + std::to_string(hidden) +
                             ", block=" + std::to_string(block) + ")");
  }
}

}  // namespace

void moe_sum_launch(void *out, const void *in, size_t tokens, size_t topk,
                    size_t hidden, int dtype_tag, void *stream) {
  using infini::rt::DataType;
  switch (static_cast<DataType>(dtype_tag)) {
    case DataType::kFloat16:
      launchMoeSum<half>(out, in, tokens, topk, hidden, stream);
      return;
    case DataType::kBFloat16:
      launchMoeSum<cuda_bfloat16>(out, in, tokens, topk, hidden, stream);
      return;
    case DataType::kFloat32:
      launchMoeSum<float>(out, in, tokens, topk, hidden, stream);
      return;
    default:
      throw std::runtime_error("QY moe_sum_launch: unsupported dtype_tag " +
                               std::to_string(dtype_tag));
  }
}

}  // namespace infini::ops::qy
