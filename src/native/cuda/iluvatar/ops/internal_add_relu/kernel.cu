#include <cuda_runtime.h>

#include "native/cuda/iluvatar/ops/internal_add_relu/kernel.cuh"

namespace infini::ops::iluvatar_add_relu_detail {

__global__ void AddReluKernel(const float* input, const float* other,
                              uint32_t count, uint32_t mode, uint32_t inner,
                              uint32_t channels, float* output) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const uint32_t j = mode == 0 ? i : (mode == 1 ? 0 : (i / inner) % channels);
    const float sum = __fadd_rn(input[i], __fmul_rn(1.0f, other[j]));
    output[i] = sum <= 0.0f ? 0.0f : sum;
  }
}

}  // namespace infini::ops::iluvatar_add_relu_detail
