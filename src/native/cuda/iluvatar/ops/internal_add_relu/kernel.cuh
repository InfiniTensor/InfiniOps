#ifndef INFINI_OPS_ILUVATAR_ADD_RELU_KERNEL_CUH_
#define INFINI_OPS_ILUVATAR_ADD_RELU_KERNEL_CUH_

#include <cstdint>

namespace infini::ops::iluvatar_add_relu_detail {

__global__ void AddReluKernel(const float* input, const float* other,
                              uint32_t count, uint32_t mode, uint32_t inner,
                              uint32_t channels, float* output);

}  // namespace infini::ops::iluvatar_add_relu_detail

#endif
