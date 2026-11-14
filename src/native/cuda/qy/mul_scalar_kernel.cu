#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "native/cuda/qy/data_type_.h"
#include "native/cuda/qy/mul_scalar_kernel.h"

namespace infini::ops::qy {
namespace {

template <typename T>
__device__ __forceinline__ T mulScalarValue(T value, double alpha) {
  return value * static_cast<T>(alpha);
}

template <>
__device__ __forceinline__ half mulScalarValue<half>(half value, double alpha) {
  return __hmul(value, __float2half(static_cast<float>(alpha)));
}

template <>
__device__ __forceinline__ cuda_bfloat16
mulScalarValue<cuda_bfloat16>(cuda_bfloat16 value, double alpha) {
  return __hmul(value, __float2bfloat16_rn(static_cast<float>(alpha)));
}

template <typename T>
__global__ void mulScalarKernel(size_t n, T *output, const T *input,
                                double alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (; idx < n; idx += stride) {
    output[idx] = mulScalarValue(input[idx], alpha);
  }
}

template <typename T>
void launchMulScalarKernel(size_t numel, void *output, const void *input,
                           double alpha, void *stream) {
  if (numel == 0) {
    return;
  }

  constexpr int block = 128;
  int grid = static_cast<int>((numel + block - 1) / block);
  // Denglin OORs on large grids even for grid-stride kernels; keep modest.
  grid = std::min(grid, 128);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  // Clear any prior sticky launch error so we attribute failures to this
  // launch.
  (void)cudaGetLastError();
  mulScalarKernel<T><<<grid, block, 0, cuda_stream>>>(
      numel, reinterpret_cast<T *>(output), reinterpret_cast<const T *>(input),
      alpha);
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("QY mul_scalar_launch failed: ") + cudaGetErrorString(err) +
        " (numel=" + std::to_string(numel) + ", grid=" + std::to_string(grid) +
        ", block=" + std::to_string(block) + ")");
  }
}

}  // namespace

void mul_scalar_launch(void *out, const void *in, size_t numel, int dtype_tag,
                       double alpha, void *stream) {
  using infini::rt::DataType;
  switch (static_cast<DataType>(dtype_tag)) {
    case DataType::kFloat16:
      launchMulScalarKernel<half>(numel, out, in, alpha, stream);
      return;
    case DataType::kBFloat16:
      launchMulScalarKernel<cuda_bfloat16>(numel, out, in, alpha, stream);
      return;
    case DataType::kFloat32:
      launchMulScalarKernel<float>(numel, out, in, alpha, stream);
      return;
    case DataType::kFloat64:
      launchMulScalarKernel<double>(numel, out, in, alpha, stream);
      return;
    default:
      throw std::runtime_error("QY mul_scalar_launch: unsupported dtype_tag " +
                               std::to_string(dtype_tag));
  }
}

}  // namespace infini::ops::qy
