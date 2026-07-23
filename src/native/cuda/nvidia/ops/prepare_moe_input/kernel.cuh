// SPDX-License-Identifier: Apache-2.0
// Adapted from SGLang at commit a2935ce329bc139701d4837ee5cd50d58898e538:
// sgl-kernel/csrc/moe/prepare_moe_input.cu

#ifndef INFINI_OPS_NVIDIA_PREPARE_MOE_INPUT_KERNEL_CUH_
#define INFINI_OPS_NVIDIA_PREPARE_MOE_INPUT_KERNEL_CUH_

#include <cstdint>

namespace infini::ops::prepare_moe_input_detail {

constexpr int32_t kBlockscaleAlignment = 128;

__global__ void PrepareMoeInputKernel(
    const int32_t* __restrict__ topk_ids, int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ problem_sizes1, int32_t* __restrict__ problem_sizes2,
    int32_t* __restrict__ input_permutation,
    int32_t* __restrict__ output_permutation,
    int32_t* __restrict__ blockscale_offsets, int64_t numel, int32_t topk,
    int32_t num_experts, int32_t n, int32_t k) {
  for (int32_t expert = threadIdx.x; expert < num_experts;
       expert += blockDim.x) {
    problem_sizes1[expert * 3] = 0;
    problem_sizes1[expert * 3 + 1] = 2 * n;
    problem_sizes1[expert * 3 + 2] = k;
    problem_sizes2[expert * 3] = 0;
    problem_sizes2[expert * 3 + 1] = k;
    problem_sizes2[expert * 3 + 2] = n;
  }
  for (int64_t i = threadIdx.x; i < numel; i += blockDim.x) {
    input_permutation[i] = -1;
    output_permutation[i] = -1;
  }
  __syncthreads();

  for (int64_t i = threadIdx.x; i < numel; i += blockDim.x) {
    const int32_t expert = topk_ids[i];
    if (expert >= 0 && expert < num_experts) {
      atomicAdd(&problem_sizes1[expert * 3], 1);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t offset = 0;
    int32_t blockscale_offset = 0;
    expert_offsets[0] = 0;
    if (blockscale_offsets != nullptr) {
      blockscale_offsets[0] = 0;
    }

    for (int32_t expert = 0; expert < num_experts; ++expert) {
      const int32_t count = problem_sizes1[expert * 3];
      problem_sizes2[expert * 3] = offset;
      offset += count;
      expert_offsets[expert + 1] = offset;

      if (blockscale_offsets != nullptr) {
        blockscale_offset += (count + kBlockscaleAlignment - 1) /
                             kBlockscaleAlignment * kBlockscaleAlignment;
        blockscale_offsets[expert + 1] = blockscale_offset;
      }
    }
  }
  __syncthreads();

  for (int64_t i = threadIdx.x; i < numel; i += blockDim.x) {
    const int32_t expert = topk_ids[i];
    if (expert >= 0 && expert < num_experts) {
      const int32_t destination = atomicAdd(&problem_sizes2[expert * 3], 1);
      input_permutation[destination] = static_cast<int32_t>(i / topk);
      output_permutation[i] = destination;
    }
  }
  __syncthreads();

  for (int32_t expert = threadIdx.x; expert < num_experts;
       expert += blockDim.x) {
    problem_sizes2[expert * 3] = problem_sizes1[expert * 3];
  }
}

}  // namespace infini::ops::prepare_moe_input_detail

#endif  // INFINI_OPS_NVIDIA_PREPARE_MOE_INPUT_KERNEL_CUH_
