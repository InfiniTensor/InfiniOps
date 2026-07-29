// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Auto-generated from vLLM generate_kernels.py at commit
// 9b9fc4039c25a6e4fe0ae97361b62edd74b8b47e for the SM80 A16 subset.
// clang-format off

#include "native/cuda/nvidia/ops/marlin_gemm/marlin_kernel.cuh"
#include "native/cuda/nvidia/ops/marlin_gemm/marlin_template.h"

namespace MARLIN_NAMESPACE_NAME {

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, true, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, true, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, true, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 2, 16, 4, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 8, 4, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 4, 8, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 3, 16, 4, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 8, 4, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 4, 8, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 4, 16, 4, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 8, 4, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 4, 8, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, true, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, true, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, true, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 2, 16, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 8, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 4, 8, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 3, 16, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 8, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 4, 8, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 4, 16, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 8, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 4, 8, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, true, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, true, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, true, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 2, 16, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 8, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 4, 8, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 3, 16, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 8, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 4, 8, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 4, 16, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 8, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 4, 8, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, true, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, true, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, true, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 2, 16, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 8, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 4, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 3, 16, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 8, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 4, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 4, 16, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 8, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 4, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);

}  // namespace MARLIN_NAMESPACE_NAME
