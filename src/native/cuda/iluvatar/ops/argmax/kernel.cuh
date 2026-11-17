#ifndef INFINI_OPS_ILUVATAR_ARGMAX_KERNEL_CUH_
#define INFINI_OPS_ILUVATAR_ARGMAX_KERNEL_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <climits>
#include <cstdint>

namespace infini::ops::iluvatar_argmax_detail {

constexpr int kThreads = 256;
template <typename T>
__device__ float ToFloat(T value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float ToFloat<half>(half value) {
  return __half2float(value);
}

template <>
__device__ inline float ToFloat<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

struct Candidate {
  float value;
  int index;
};

static_assert(sizeof(Candidate) == 8);

// The frozen PyTorch Argmax contract prefers NaN over finite values and the
// first index among equal maxima (including multiple NaNs).
__device__ inline Candidate Better(Candidate a, Candidate b) {
  if (b.index == INT_MAX) return a;
  if (a.index == INT_MAX) return b;
  const bool a_nan = isnan(a.value);
  const bool b_nan = isnan(b.value);
  if (a_nan != b_nan) return b_nan ? b : a;
  if (a_nan || a.value == b.value) return b.index < a.index ? b : a;
  return b.value > a.value ? b : a;
}

template <typename T>
__global__ void RowsKernel(const T* input, std::int64_t* output, int width) {
  const int row = blockIdx.x;
  const T* values = input + static_cast<std::size_t>(row) * width;
  Candidate best{-CUDART_INF_F, INT_MAX};
  for (std::int64_t column = threadIdx.x; column < width;
       column += blockDim.x) {
    best = Better(best, {ToFloat(values[column]), static_cast<int>(column)});
  }
  __shared__ Candidate partial[kThreads];
  partial[threadIdx.x] = best;
  __syncthreads();
  for (int offset = kThreads / 2; offset; offset /= 2) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] =
          Better(partial[threadIdx.x], partial[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output[row] = partial[0].index;
}

}  // namespace infini::ops::iluvatar_argmax_detail

#endif  // INFINI_OPS_ILUVATAR_ARGMAX_KERNEL_CUH_
