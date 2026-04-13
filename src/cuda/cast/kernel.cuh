#ifndef INFINI_OPS_CUDA_CAST_KERNEL_CUH_
#define INFINI_OPS_CUDA_CAST_KERNEL_CUH_

#include "cuda/kernel_commons.cuh"

namespace infini::ops {

template <Device::Type kDev, typename InT, typename OutT,
          unsigned int BLOCK_SIZE>
__global__ void CastKernel(OutT* __restrict__ out,
                           const InT* __restrict__ input,
                           const size_t* __restrict__ out_shape,
                           const size_t* __restrict__ input_shape,
                           const ptrdiff_t* __restrict__ out_strides,
                           const ptrdiff_t* __restrict__ input_strides,
                           size_t output_size, size_t ndim, bool out_contiguous,
                           bool input_contiguous) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < output_size) {
    size_t out_idx =
        out_contiguous ? idx : IndexToOffset(idx, ndim, out_shape, out_strides);
    size_t input_idx =
        input_contiguous ? idx
                         : IndexToOffset(idx, ndim, input_shape, input_strides);

    out[out_idx] = Caster<kDev>::template Cast<OutT>(input[input_idx]);
  }
}

}  // namespace infini::ops

#endif
