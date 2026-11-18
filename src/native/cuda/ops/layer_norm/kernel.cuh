#ifndef INFINI_OPS_CUDA_LAYER_NORM_KERNEL_CUH_
#define INFINI_OPS_CUDA_LAYER_NORM_KERNEL_CUH_

#include <cstddef>
#include <cub/block/block_reduce.cuh>

#include "native/cuda/caster.cuh"

namespace infini::ops {

struct CudaLayerNormLayout {
  std::size_t shape[8];
  std::size_t input_strides[8];
  std::size_t out_strides[8];
  std::size_t weight_stride;
  std::size_t bias_stride;
  std::size_t dim;
  unsigned rank;
};

template <unsigned int block_size, Device::Type device_type, typename T>
__global__ void CudaLayerNormKernel(const T* input, const T* weight,
                                    const T* bias, T* out,
                                    CudaLayerNormLayout layout, float eps) {
  std::size_t row = blockIdx.x;
  std::size_t input_offset = 0;
  std::size_t out_offset = 0;
  for (unsigned i = layout.rank - 1; i > 0; --i) {
    const unsigned axis = i - 1;
    const std::size_t coordinate = row % layout.shape[axis];
    row /= layout.shape[axis];
    input_offset += coordinate * layout.input_strides[axis];
    out_offset += coordinate * layout.out_strides[axis];
  }

  using Reduce = cub::BlockReduce<float, block_size>;
  __shared__ typename Reduce::TempStorage reduce_storage;
  __shared__ float mean;
  __shared__ float inv_std;

  auto load = [&](std::size_t index) {
    return Caster<device_type>::template Cast<float>(
        input[input_offset + index * layout.input_strides[layout.rank - 1]]);
  };

  float partial = 0.0f;
  for (std::size_t d = threadIdx.x; d < layout.dim; d += block_size) {
    partial += load(d);
  }
  const float sum = Reduce(reduce_storage).Sum(partial);
  if (threadIdx.x == 0) mean = sum / static_cast<float>(layout.dim);
  __syncthreads();

  partial = 0.0f;
  for (std::size_t d = threadIdx.x; d < layout.dim; d += block_size) {
    const float centered = load(d) - mean;
    partial += centered * centered;
  }
  const float variance_sum = Reduce(reduce_storage).Sum(partial);
  if (threadIdx.x == 0) {
    inv_std = rsqrtf(variance_sum / static_cast<float>(layout.dim) + eps);
  }
  __syncthreads();

  for (std::size_t d = threadIdx.x; d < layout.dim; d += block_size) {
    const float scale = Caster<device_type>::template Cast<float>(
        weight[d * layout.weight_stride]);
    const float shift = bias ? Caster<device_type>::template Cast<float>(
                                   bias[d * layout.bias_stride])
                             : 0.0f;
    out[out_offset + d * layout.out_strides[layout.rank - 1]] =
        Caster<device_type>::template Cast<T>(
            (load(d) - mean) * inv_std * scale + shift);
  }
}

}  // namespace infini::ops

#endif
