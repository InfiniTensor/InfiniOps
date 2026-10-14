#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "native/cuda/qy/data_type_.h"
#include "native/cuda/qy/silu_and_mul_kernel.h"

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

template <typename T>
__global__ void siluAndMulKernel(T *y, const T *x, size_t n, size_t hidden) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (; idx < n; idx += stride) {
    size_t row = idx / hidden;
    size_t col = idx - row * hidden;
    const T *row_x = x + row * hidden * 2;
    float gate = toFloat(row_x[col]);
    float up = toFloat(row_x[col + hidden]);
    float silu = gate / (1.0f + expf(-gate));
    y[idx] = fromFloat<T>(silu * up);
  }
}

template <typename T>
void launchSiluAndMul(void *out, const void *in, size_t rows, size_t hidden,
                      void *stream) {
  size_t n = rows * hidden;
  if (n == 0) {
    return;
  }

  constexpr int block = 128;
  int grid = static_cast<int>((n + block - 1) / block);
  grid = std::min(grid, 128);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  (void)cudaGetLastError();
  siluAndMulKernel<T><<<grid, block, 0, cuda_stream>>>(
      reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in), n, hidden);
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("QY silu_and_mul_launch failed: ") +
                             cudaGetErrorString(err) +
                             " (n=" + std::to_string(n) +
                             ", rows=" + std::to_string(rows) +
                             ", hidden=" + std::to_string(hidden) +
                             ", grid=" + std::to_string(grid) +
                             ", block=" + std::to_string(block) + ")");
  }
}

}  // namespace

void silu_and_mul_launch(void *out, const void *in, size_t rows, size_t hidden,
                         int dtype_tag, void *stream) {
  using infini::rt::DataType;
  switch (static_cast<DataType>(dtype_tag)) {
    case DataType::kFloat16:
      launchSiluAndMul<half>(out, in, rows, hidden, stream);
      return;
    case DataType::kBFloat16:
      launchSiluAndMul<cuda_bfloat16>(out, in, rows, hidden, stream);
      return;
    case DataType::kFloat32:
      launchSiluAndMul<float>(out, in, rows, hidden, stream);
      return;
    default:
      throw std::runtime_error(
          "QY silu_and_mul_launch: unsupported dtype_tag " +
          std::to_string(dtype_tag));
  }
}

}  // namespace infini::ops::qy
