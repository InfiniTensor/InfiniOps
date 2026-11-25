#ifndef INFINI_OPS_METAX_FUSED_ADD_RMS_NORM_KERNEL_CUH_
#define INFINI_OPS_METAX_FUSED_ADD_RMS_NORM_KERNEL_CUH_

#include <cub/block/block_reduce.cuh>

#include "native/cuda/metax/caster.cuh"

namespace infini::ops {

template <typename T>
struct alignas(16) MetaxNormPack {
  T values[8];
};

template <typename T>
__global__ void MetaxFusedAddRmsNormVectorizedKernel(
    T* input, int64_t input_stride, T* residual, int64_t residual_stride,
    const T* weight, float epsilon) {
  constexpr int kThreads = 256;
  constexpr int kItems = 16;
  constexpr auto kDevice = Device::Type::kMetax;
  using Pack = MetaxNormPack<T>;
  auto x = reinterpret_cast<Pack*>(input + blockIdx.x * input_stride);
  auto r = reinterpret_cast<Pack*>(residual + blockIdx.x * residual_stride);
  auto w = reinterpret_cast<const Pack*>(weight);
  float values[kItems];
  float sum_squared = 0.0f;

#pragma unroll
  for (int pack = 0; pack < 2; ++pack) {
    const int offset = threadIdx.x * 2 + pack;
    Pack xv = x[offset];
    Pack rv = r[offset];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float merged = Caster<kDevice>::template Cast<float>(xv.values[i]) +
                     Caster<kDevice>::template Cast<float>(rv.values[i]);
      rv.values[i] = Caster<kDevice>::template Cast<T>(merged);
      float rounded = Caster<kDevice>::template Cast<float>(rv.values[i]);
      values[pack * 8 + i] = rounded;
      sum_squared += rounded * rounded;
    }
    r[offset] = rv;
  }

  using Reduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename Reduce::TempStorage storage;
  float total = Reduce(storage).Sum(sum_squared);
  __shared__ float inverse_rms;
  if (threadIdx.x == 0) {
    inverse_rms = rsqrtf(total / 4096.0f + epsilon);
  }
  __syncthreads();

#pragma unroll
  for (int pack = 0; pack < 2; ++pack) {
    const int offset = threadIdx.x * 2 + pack;
    Pack out;
    Pack gamma;
    if (weight != nullptr) {
      gamma = w[offset];
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float value = values[pack * 8 + i] * inverse_rms;
      if (weight != nullptr) {
        value *= Caster<kDevice>::template Cast<float>(gamma.values[i]);
      }
      out.values[i] = Caster<kDevice>::template Cast<T>(value);
    }
    x[offset] = out;
  }
}

}  // namespace infini::ops

#endif
