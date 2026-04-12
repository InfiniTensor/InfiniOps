#ifndef INFINI_OPS_CUDA_LINEAR_KERNEL_CUH_
#define INFINI_OPS_CUDA_LINEAR_KERNEL_CUH_

#include "cuda/kernel_commons.cuh"

namespace infini::ops {

template <typename T>
__global__ void BiasAddKernel(T* out, const T* bias, size_t rows, size_t cols) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < rows * cols) {
    size_t col = idx % cols;
    out[idx] += bias[col];
  }
}

}  // namespace infini::ops

#endif
