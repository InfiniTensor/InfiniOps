#include <cuda_fp16.h>

#include <cfloat>
#include <cstdint>
#include <stdexcept>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/ops/argmax/kernel.h"

namespace infini::ops::mars_argmax_detail {
namespace {

__host__ __device__ inline float ToFloat(float value) { return value; }
__host__ __device__ inline float ToFloat(__half value) {
  return __half2float(value);
}
__host__ __device__ inline float ToFloat(__hpcc_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename TData, typename TIndex>
__global__ void Argmax1DKernel(const TData* input, std::size_t count,
                               TIndex* out) {
  constexpr int kThreads = 256;
  __shared__ float values[kThreads];
  __shared__ std::size_t indices[kThreads];

  const int tid = static_cast<int>(threadIdx.x);
  float best = -FLT_MAX;
  std::size_t best_index = 0;
  for (std::size_t i = tid; i < count; i += kThreads) {
    const float value = ToFloat(input[i]);
    if (value > best || (value == best && i < best_index)) {
      best = value;
      best_index = i;
    }
  }
  values[tid] = best;
  indices[tid] = best_index;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      const float other = values[tid + stride];
      const std::size_t other_index = indices[tid + stride];
      if (other > values[tid] ||
          (other == values[tid] && other_index < indices[tid])) {
        values[tid] = other;
        indices[tid] = other_index;
      }
    }
    __syncthreads();
  }

  if (tid == 0) *out = static_cast<TIndex>(indices[0]);
}

template <typename TData>
void Submit(const void* input, std::size_t count, void* out, DataType out_dtype,
            cudaStream_t stream) {
  if (out_dtype == DataType::kInt64)
    Argmax1DKernel<TData, std::int64_t>
        <<<1, 256, 0, stream>>>(static_cast<const TData*>(input), count,
                                static_cast<std::int64_t*>(out));
  else
    Argmax1DKernel<TData, std::int32_t>
        <<<1, 256, 0, stream>>>(static_cast<const TData*>(input), count,
                                static_cast<std::int32_t*>(out));
}

}  // namespace

void Launch(const void* input, DataType dtype, std::size_t count, void* out,
            DataType out_dtype, cudaStream_t stream) {
  if (!count)
    throw std::invalid_argument("Mars Argmax requires a nonempty input");
  switch (dtype) {
    case DataType::kFloat16:
      Submit<__half>(input, count, out, out_dtype, stream);
      break;
    case DataType::kBFloat16:
      Submit<__hpcc_bfloat16>(input, count, out, out_dtype, stream);
      break;
    case DataType::kFloat32:
      Submit<float>(input, count, out, out_dtype, stream);
      break;
    default:
      throw std::invalid_argument("Mars Argmax unsupported dtype");
  }
}

}  // namespace infini::ops::mars_argmax_detail
