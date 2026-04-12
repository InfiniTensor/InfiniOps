#ifndef INFINI_OPS_CUDA_CAT_KERNEL_CUH_
#define INFINI_OPS_CUDA_CAT_KERNEL_CUH_

#include "cuda/kernel_commons.cuh"

namespace infini::ops {

template <typename T>
__global__ void CatKernel(T* __restrict__ out,
                          const void* const* __restrict__ inputs,
                          const size_t* __restrict__ cum_sizes,
                          size_t input_count, size_t outer_size,
                          size_t inner_size, size_t total_dim_size,
                          size_t output_size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= output_size) {
    return;
  }

  // Decompose flat index into (outer, dim_and_inner).
  size_t slice_size = total_dim_size * inner_size;
  size_t outer = idx / slice_size;
  size_t rem = idx % slice_size;
  size_t dim_idx = rem / inner_size;
  size_t inner = rem % inner_size;

  // Find which input tensor this element belongs to via cumulative sizes.
  size_t input_idx = 0;

  for (size_t i = 0; i < input_count; ++i) {
    if (dim_idx < cum_sizes[i]) {
      input_idx = i;
      break;
    }
  }

  // Compute the local dimension index within the input tensor.
  size_t local_dim = dim_idx - (input_idx > 0 ? cum_sizes[input_idx - 1] : 0);
  size_t input_dim_size =
      cum_sizes[input_idx] - (input_idx > 0 ? cum_sizes[input_idx - 1] : 0);

  const T* in_ptr = static_cast<const T*>(inputs[input_idx]);
  size_t in_offset = outer * input_dim_size * inner_size +
                     local_dim * inner_size + inner;

  out[idx] = in_ptr[in_offset];
}

}  // namespace infini::ops

#endif
