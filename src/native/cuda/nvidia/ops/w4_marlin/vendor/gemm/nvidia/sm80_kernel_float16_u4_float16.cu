// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project.
// Generated from InfiniCore 1ab85ef1d3c2bce49edf1fc73a5b48becaf7552e,
// src/infiniop/ops/awq_marlin_gemm/nvidia/generate_kernels.py.
// Modified for InfiniOps: W4A16-only instantiations and relative includes.
// See repository THIRD_PARTY.txt and LICENSE.
#include "../marlin/kernel.h"
#include "../marlin/marlin_template.h"
namespace MARLIN_NAMESPACE_NAME {
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 4, -1, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 4, 4, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
}
