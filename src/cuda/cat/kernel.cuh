#ifndef INFINI_OPS_CUDA_CAT_KERNEL_CUH_
#define INFINI_OPS_CUDA_CAT_KERNEL_CUH_

#include "cuda/kernel_commons.cuh"

namespace infini::ops {

template <typename T, unsigned int BLOCK_SIZE>
__global__ void CatKernel(T* __restrict__ out,
                          const T* const* __restrict__ inputs,
                          const size_t* __restrict__ input_dim_sizes,
                          const size_t* __restrict__ input_dim_offsets,
                          size_t input_count, size_t out_dim_size, size_t inner,
                          size_t output_size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < output_size) {
    size_t outer_idx = idx / (out_dim_size * inner);
    size_t rem = idx % (out_dim_size * inner);
    size_t dim_idx = rem / inner;
    size_t inner_idx = rem % inner;

    size_t input_idx = 0;
    while (input_idx + 1 < input_count &&
           dim_idx >= input_dim_offsets[input_idx + 1]) {
      ++input_idx;
    }

    size_t local_dim_idx = dim_idx - input_dim_offsets[input_idx];
    size_t src_idx =
        (outer_idx * input_dim_sizes[input_idx] + local_dim_idx) * inner +
        inner_idx;

    out[idx] = inputs[input_idx][src_idx];
  }
}

}  // namespace infini::ops

#endif
