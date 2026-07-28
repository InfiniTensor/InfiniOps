// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from vLLM at commit 9b9fc4039c25a6e4fe0ae97361b62edd74b8b47e:
// csrc/libtorch_stable/quantization/marlin/marlin.cu

/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME infini::ops::marlin_gemm_detail
#endif

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdint>
#include <type_traits>

#include "native/cuda/nvidia/ops/marlin_gemm/kernel.h"
#include "native/cuda/nvidia/ops/marlin_gemm/marlin_kernel.cuh"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

#define MARLIN_CHECK(condition, ...) assert(condition)

namespace infini::ops::marlin_gemm_detail {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
__global__ void permute_cols_kernel(int4 const* __restrict__ a_int4_ptr,
                                    int const* __restrict__ perm_int_ptr,
                                    int4* __restrict__ out_int4_ptr, int size_m,
                                    int size_k, int lda, int block_rows) {
  auto start_row = block_rows * blockIdx.x;
  int finish_row = start_row + block_rows;
  if (finish_row > size_m) {
    finish_row = size_m;
  }
  int cur_block_rows = finish_row - start_row;

  int input_row_stride = lda * sizeof(half) / 16;
  int output_row_stride = size_k * sizeof(half) / 16;

  auto permute_row = [&](int row) {
    int iters = size_k / default_threads;
    int rest = size_k % default_threads;

    int input_offset = row * input_row_stride;
    int output_offset = row * output_row_stride;

    half const* a_row_half =
        reinterpret_cast<half const*>(a_int4_ptr + input_offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + output_offset);

    int base_k = 0;

    for (int i = 0; i < iters; i++) {
      auto cur_k = base_k + threadIdx.x;
      int src_pos = perm_int_ptr[cur_k];

      out_half[cur_k] = a_row_half[src_pos];

      base_k += default_threads;
    }

    if (rest) {
      if (threadIdx.x < rest) {
        auto cur_k = base_k + threadIdx.x;
        int src_pos = perm_int_ptr[cur_k];

        out_half[cur_k] = a_row_half[src_pos];
      }
    }
  };

  for (int i = 0; i < cur_block_rows; i++) {
    int cur_row = start_row + i;
    if (cur_row < size_m) {
      permute_row(cur_row);
    }
  }
}

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          bool has_act_order, bool is_k_full, int stages) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups =
        tb_groups * stages * 2;          // Chunk size is 2x pipeline over dim K
    load_groups = max(load_groups, 32);  // We load at least 32 scale groups
    return load_groups * tb_n * 2;
  } else {
    int tb_scales = tb_groups * tb_n * 2;

    return tb_scales * stages;
  }
}

int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks,
                          int prob_m, int prob_n, int prob_k, int num_bits,
                          int group_size, bool has_act_order, bool is_k_full,
                          int has_zp, bool is_zp_float, bool is_a_8bit,
                          int stages) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;
  int sh_a_size = stages * (tb_m * tb_k) * (is_a_8bit ? 1 : 2);
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size =
      (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits,
                            group_size, has_act_order, is_k_full, stages);
  int sh_g_idx_size = has_act_order && !is_k_full ? stages * tb_k / 4 : 0;
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size =
      tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size;

  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, bool has_act_order, bool is_k_full,
                     int has_zp, bool is_zp_float, bool is_a_8bit, int stages,
                     int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  // Check that pipeline fits into cache
  int cache_size = get_kernel_cache_size(
      th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits, group_size,
      has_act_order, is_k_full, has_zp, is_zp_float, is_a_8bit, stages);
  return cache_size <= max_shared_mem;
}

MarlinFuncPtr get_marlin_kernel(
    const vllm::ScalarType a_type, const vllm::ScalarType b_type,
    const vllm::ScalarType c_type, const vllm::ScalarType s_type,
    int thread_m_blocks, int thread_n_blocks, int thread_k_blocks,
    bool m_block_size_8, bool has_act_order, bool has_zp, int group_blocks,
    int threads, bool is_zp_float, int stages) {
  int num_bits = b_type.size_bits();
  auto kernel = MarlinDefault;

#include "native/cuda/nvidia/ops/marlin_gemm/kernel_selector.h"

  return kernel;
}

exec_config_t determine_exec_config(
    const vllm::ScalarType& a_type, const vllm::ScalarType& b_type,
    const vllm::ScalarType& c_type, const vllm::ScalarType& s_type, int prob_m,
    int prob_n, int prob_k, int thread_m_blocks, bool m_block_size_8,
    int num_bits, int group_size, bool has_act_order, bool is_k_full,
    bool has_zp, bool is_zp_float, int is_a_8bit, int stages,
    int max_shared_mem, int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1
                                        ? large_batch_thread_configs
                                        : small_batch_thread_configs;
  int thread_configs_size =
      thread_m_blocks > 1
          ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
          : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(th_config, thread_m_blocks, prob_m, prob_n, prob_k,
                         num_bits, group_size, has_act_order, is_k_full, has_zp,
                         is_zp_float, is_a_8bit, stages,
                         max_shared_mem - 512)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(th_config, thread_m_blocks, prob_m,
                                           prob_n, prob_k, num_bits, group_size,
                                           has_act_order, is_k_full, has_zp,
                                           is_zp_float, is_a_8bit, stages);

    int group_blocks = 0;
    if (!has_act_order) {
      group_blocks = group_size == -1 ? -1 : group_size / 16;
    }

    auto kernel =
        get_marlin_kernel(a_type, b_type, c_type, s_type, thread_m_blocks,
                          th_config.thread_n / 16, th_config.thread_k / 16,
                          m_block_size_8, has_act_order, has_zp, group_blocks,
                          th_config.num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) continue;

    return {1, th_config};
  }

  return exec_cfg;
}

void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, const void* b_s, void* g_s, void* zp, void* g_idx,
               void* perm, void* a_tmp, int prob_m, int prob_n, int prob_k,
               int lda, void* workspace, vllm::ScalarType const& a_type,
               vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
               vllm::ScalarType const& s_type, bool has_bias,
               bool has_act_order, bool is_k_full, bool has_zp, int num_groups,
               int group_size, int dev, cudaStream_t stream, int thread_k_init,
               int thread_n_init, int sms, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float) {
  bool is_a_8bit = a_type.size_bits() == 8;
  MARLIN_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [",
               prob_m, ", ", prob_n, ", ", prob_k, "]");

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      MARLIN_CHECK(group_size != -1);
      group_blocks = group_size / 16;
      MARLIN_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                   " is not divisible by group_blocks = ", group_blocks);
    } else {
      MARLIN_CHECK(group_size == 0);
      group_blocks = 0;
    }
  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      MARLIN_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                   " is not divisible by group_blocks = ", group_blocks);
    }
  }

  int num_bits = b_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;

  const int4* bias_ptr = (const int4*)b_bias;
  const float* a_s_ptr = (const float*)a_s;
  const int4* b_s_ptr = (const int4*)b_s;
  const float* g_s_ptr = (const float*)g_s;

  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = (const int*)g_idx;
  const int* perm_ptr = (const int*)perm;
  int4* a_tmp_ptr = (int4*)a_tmp;
  int* locks = (int*)workspace;

  if (has_act_order) {
    // Permute A columns
    int block_rows = div_ceil(prob_m, sms);
    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    permute_cols_kernel<<<sms, default_threads, 0, stream>>>(
        A_ptr, perm_ptr, a_tmp_ptr, prob_m, prob_k, lda, block_rows);
    // clang-format on
    A_ptr = a_tmp_ptr;
    lda = prob_k;

    // If we have a full K, then we can run the non-act-order version of Marlin
    // (since the weight rows are reordered by increasing group ids, and by
    // having a full K, we have full original groups)
    if (is_k_full) has_act_order = false;
  }

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  MARLIN_CHECK(max_shared_mem > 0);

  int major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         dev);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         dev);
  MARLIN_CHECK(major_capability * 10 + minor_capability >= 75,
               "marlin kernel only support Turing or newer GPUs.");
  int stages = 4;
  if (major_capability == 7 && minor_capability == 5) {
    stages = 2;
    MARLIN_CHECK(a_type == vllm::kFloat16 || a_type == vllm::kS8,
                 "Turing only support FP16 or INT8 activation.");
  }
  if (a_type == vllm::kFE4M3fn) {
    MARLIN_CHECK(major_capability * 10 + minor_capability >= 89,
                 "FP8 only support Ada Lovelace or newer GPUs.");
    MARLIN_CHECK(
        major_capability * 10 + minor_capability == 89 ||
            major_capability == 12,
        "Marlin W4A8-FP8 only support SM89 or SM12x device (It is slower than "
        "Marlin W4A16 on other devices).");
  }

  int max_par = 16;
  if (prob_n <= 4096) max_par = 16 * 8;
  int max_shared_mem_new = max_shared_mem;
  int rest_m = prob_m;
  int max_thread_m_blocks = 4;
  while (rest_m) {
    int par_count = rest_m / (max_thread_m_blocks * 16);
    if (par_count > max_par) par_count = max_par;
    int prob_m_split =
        par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

    int thread_k = thread_k_init;
    int thread_n = thread_n_init;

    int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    int m_block_size_8 = prob_m_split <= 8 && a_type.size_bits() == 16;

    // Set thread config
    exec_config_t exec_cfg;
    thread_config_t thread_tfg;
    if (thread_k != -1 && thread_n != -1) {
      thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
      exec_cfg = exec_config_t{1, thread_tfg};
      MARLIN_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
                   " is not divisible by thread_n = ", thread_n);
      MARLIN_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
                   " is not divisible by thread_k = ", thread_k);
    } else {
      // Auto config
      exec_cfg = determine_exec_config(
          a_type, b_type, c_type, s_type, prob_m_split, prob_n, prob_k,
          thread_m_blocks, m_block_size_8, num_bits, group_size, has_act_order,
          is_k_full, has_zp, is_zp_float, is_a_8bit, stages, max_shared_mem,
          sms);
      thread_tfg = exec_cfg.tb_cfg;
      if (thread_tfg.thread_n != -1) {
        if (prob_n / thread_tfg.thread_n *
                div_ceil(prob_m_split, thread_m_blocks * 16) * 4 <=
            sms) {
          if (is_valid_config({128, 64, 128}, thread_m_blocks, prob_m_split,
                              prob_n, prob_k, num_bits, group_size,
                              has_act_order, is_k_full, has_zp, is_zp_float,
                              is_a_8bit, stages, max_shared_mem_new)) {
            thread_tfg = {128, 64, 128};
            exec_cfg = {1, thread_tfg};
          }
        }
      }

      if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1) {
        max_thread_m_blocks--;
        continue;
      }
    }

    int num_threads = thread_tfg.num_threads;
    thread_k = thread_tfg.thread_k;
    thread_n = thread_tfg.thread_n;
    int blocks = sms * exec_cfg.blocks_per_sm;
    if (exec_cfg.blocks_per_sm > 1)
      max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    MARLIN_CHECK(
        is_valid_config(thread_tfg, thread_m_blocks, prob_m_split, prob_n,
                        prob_k, num_bits, group_size, has_act_order, is_k_full,
                        has_zp, is_zp_float, is_a_8bit, stages,
                        max_shared_mem_new),
        "Invalid thread config: thread_m_blocks = ", thread_m_blocks,
        ", thread_k = ", thread_tfg.thread_k,
        ", thread_n = ", thread_tfg.thread_n,
        ", num_threads = ", thread_tfg.num_threads, " for MKN = [", prob_m,
        ", ", prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
        ", prob_m_split = ", prob_m_split, ", group_size = ", group_size,
        ", has_act_order = ", has_act_order, ", is_k_full = ", is_k_full,
        ", has_zp = ", has_zp, ", is_zp_float = ", is_zp_float,
        ", stages = ", stages, ", max_shared_mem_new = ", max_shared_mem_new);

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
        thread_k_blocks, m_block_size_8, has_act_order, has_zp, group_blocks,
        num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) {
      MARLIN_CHECK(false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n,
                   ", ", prob_k, "]", ", has_act_order = ", has_act_order,
                   ", num_groups = ", num_groups, ", group_size = ", group_size,
                   ", prob_m_split = ", prob_m_split,
                   ", thread_m_blocks = ", thread_m_blocks,
                   ", thread_n_blocks = ", thread_n_blocks,
                   ", thread_k_blocks = ", thread_k_blocks,
                   ", num_threads = ", num_threads, ", num_bits = ", num_bits);
    }

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_new);

    bool part_use_atomic_add =
        use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    kernel<<<blocks, num_threads, max_shared_mem_new, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr,
        g_idx_ptr, num_groups,
        prob_m_split, prob_n, prob_k, lda, locks, has_bias, part_use_atomic_add,
        use_fp32_reduce, max_shared_mem_new);
    // clang-format on

    bool is_a_8bit = a_type.size_bits() == 8;
    A_ptr += prob_m_split * (lda / (is_a_8bit ? 16 : 8));
    if (is_a_8bit) {
      a_s_ptr += prob_m_split;
    }
    C_ptr += prob_m_split * (prob_n / 8);
    rest_m -= prob_m_split;
  }
}

}  // namespace infini::ops::marlin_gemm_detail

#undef MARLIN_CHECK

namespace infini::ops {
namespace {

class DeviceGuard {
 public:
  explicit DeviceGuard(int device_index) {
    auto status = cudaGetDevice(&previous_device_);
    assert(status == cudaSuccess &&
           "`MarlinGemm` failed to query the current CUDA device");
    if (previous_device_ != device_index) {
      status = cudaSetDevice(device_index);
      assert(status == cudaSuccess &&
             "`MarlinGemm` failed to select the input CUDA device");
      restore_ = true;
    }
  }

  ~DeviceGuard() {
    if (restore_) {
      const auto status = cudaSetDevice(previous_device_);
      assert(status == cudaSuccess &&
             "`MarlinGemm` failed to restore the CUDA device");
    }
  }

 private:
  int previous_device_{0};
  bool restore_{false};
};

vllm::ScalarType ActivationType(DataType dtype) {
  if (dtype == DataType::kFloat16) {
    return vllm::kFloat16;
  }
  assert(dtype == DataType::kBFloat16 &&
         "`MarlinGemm` NVIDIA backend only supports A16 activations");
  return vllm::kBFloat16;
}

bool IsSupportedWeightType(const vllm::ScalarType& type) {
  return type == vllm::kU4 || type == vllm::kU4B8 || type == vllm::kU8B128;
}

}  // namespace

Operator<MarlinGemm, Device::Type::kNvidia, 0>::Operator(
    const Tensor a, const Tensor b_q_weight, std::optional<Tensor> b_bias,
    const Tensor b_scales, std::optional<Tensor> a_scales,
    std::optional<Tensor> global_scale, std::optional<Tensor> b_zeros,
    std::optional<Tensor> g_idx, std::optional<Tensor> perm,
    const Tensor workspace, int64_t b_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float, Tensor out)
    : MarlinGemm(a, b_q_weight, b_bias, b_scales, a_scales, global_scale,
                 b_zeros, g_idx, perm, workspace, b_type_id, size_m, size_n,
                 size_k, is_k_full, use_atomic_add, use_fp32_reduce,
                 is_zp_float, out) {
  const auto b_type = vllm::ScalarType::from_id(b_type_id_);
  assert(IsSupportedWeightType(b_type) &&
         "`MarlinGemm` NVIDIA backend supports U4, U4B8, and U8B128 weights");
  assert(
      (a.dtype() == DataType::kFloat16 || a.dtype() == DataType::kBFloat16) &&
      a.dtype() == out.dtype() && b_scales.dtype() == out.dtype() &&
      "`MarlinGemm` NVIDIA backend requires matching A16 output/scales");
  assert(reinterpret_cast<uintptr_t>(a.data()) % 16 == 0 &&
         "`MarlinGemm` requires a 16-byte aligned activation pointer");
  assert(!a_scales && !global_scale && !is_zp_float_ &&
         "`MarlinGemm` NVIDIA A16 backend does not accept activation or "
         "global scales or floating zero points");

  const auto bits = b_type.size_bits();
  const auto pack_factor = 32 / bits;
  assert(b_q_weight.size(1) ==
             static_cast<Tensor::Size>(size_n_ * 16 / pack_factor) &&
         "`MarlinGemm` received an incompatible packed weight shape");

  const auto has_zp = b_zeros && b_zeros->numel() > 0;
  assert(
      ((b_type == vllm::kU4 && has_zp) || (b_type != vllm::kU4 && !has_zp)) &&
      "`MarlinGemm` U4 requires packed zero points; biased formats do not");
  if (has_zp) {
    assert(b_zeros->dtype() == DataType::kInt32 &&
           b_zeros->size(1) ==
               static_cast<Tensor::Size>(size_n_ / pack_factor) &&
           "`MarlinGemm` received an incompatible packed zero-point shape");
  }

  const auto has_act_order = g_idx && g_idx->numel() > 0;
  assert(!(b_type == vllm::kU4 && has_act_order && !is_k_full_) &&
         "`MarlinGemm` U4 act-order requires a full K reduction");
  int64_t group_size = -1;
  if (has_act_order && !is_k_full_) {
    group_size = 0;
  } else if (num_groups_ > 1) {
    assert(size_k_ % static_cast<int64_t>(num_groups_) == 0 &&
           "`MarlinGemm` requires scale groups to divide K");
    group_size = size_k_ / static_cast<int64_t>(num_groups_);
  }
  assert((group_size == -1 || group_size == 0 || group_size == 32 ||
          group_size == 64 || group_size == 128) &&
         "`MarlinGemm` received an unsupported group size");

  cudaDeviceProp properties{};
  const auto status = cudaGetDeviceProperties(&properties, device_index_);
  assert(status == cudaSuccess && properties.major >= 8 &&
         "`MarlinGemm` SM80 kernels require compute capability 8.0 or newer");
  sms_ = properties.multiProcessorCount;
  assert(workspace.numel() >= static_cast<Tensor::Size>(sms_) &&
         "`MarlinGemm` workspace must contain at least one lock per SM");
}

void Operator<MarlinGemm, Device::Type::kNvidia, 0>::operator()(
    const Tensor a, const Tensor b_q_weight, std::optional<Tensor> b_bias,
    const Tensor b_scales, std::optional<Tensor> a_scales,
    std::optional<Tensor> global_scale, std::optional<Tensor> b_zeros,
    std::optional<Tensor> g_idx, std::optional<Tensor> perm,
    const Tensor workspace, int64_t b_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float, Tensor out) const {
  ValidateCallMetadata(a, b_q_weight, b_bias, b_scales, a_scales, global_scale,
                       b_zeros, g_idx, perm, workspace, b_type_id, size_m,
                       size_n, size_k, is_k_full, use_atomic_add,
                       use_fp32_reduce, is_zp_float, out);
  if (size_m_ == 0) {
    return;
  }

  DeviceGuard guard(device_index_);
  const auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : nullptr);
  const auto a_type = ActivationType(a.dtype());
  const auto b_type = vllm::ScalarType::from_id(b_type_id_);
  const auto has_bias = b_bias.has_value();
  const auto has_zp = b_zeros && b_zeros->numel() > 0;
  const auto has_act_order = g_idx && g_idx->numel() > 0;

  int group_size = -1;
  if (has_act_order && !is_k_full_) {
    group_size = 0;
  } else if (num_groups_ > 1) {
    group_size = static_cast<int>(size_k_ / static_cast<int64_t>(num_groups_));
  }

  void* a_tmp = nullptr;
  void* c_tmp = nullptr;
  auto status = cudaSuccess;
  if (has_act_order) {
    status = cudaMallocAsync(&a_tmp, static_cast<size_t>(size_m_ * size_k_) * 2,
                             stream);
    assert(status == cudaSuccess &&
           "`MarlinGemm` failed to allocate the act-order buffer");
  }
  if (use_fp32_reduce_) {
    const auto rounded_m = ((size_m_ + 15) / 16) * 16;
    const auto reduce_rows = rounded_m < 64 ? rounded_m : 64;
    const auto c_tmp_size = static_cast<size_t>(sms_) * reduce_rows *
                            marlin_gemm_detail::max_thread_n * sizeof(float);
    status = cudaMallocAsync(&c_tmp, c_tmp_size, stream);
    assert(status == cudaSuccess &&
           "`MarlinGemm` failed to allocate the FP32 reduction buffer");
  }

  // vLLM exposes the lock workspace as an input even though Marlin updates and
  // restores its lock words during execution.
  auto* workspace_ptr = const_cast<void*>(workspace.data());

  marlin_gemm_detail::marlin_mm(
      a.data(), b_q_weight.data(), out.data(), c_tmp,
      b_bias ? b_bias->data() : nullptr, nullptr, b_scales.data(), nullptr,
      b_zeros ? b_zeros->data() : nullptr, g_idx ? g_idx->data() : nullptr,
      perm ? perm->data() : nullptr, a_tmp, static_cast<int>(size_m_),
      static_cast<int>(size_n_), static_cast<int>(size_k_),
      static_cast<int>(a.stride(0)), workspace_ptr, a_type, b_type, a_type,
      a_type, has_bias, has_act_order, is_k_full_, has_zp,
      static_cast<int>(num_groups_), group_size, device_index_, stream, -1, -1,
      sms_, use_atomic_add_, use_fp32_reduce_, false);

  status = cudaGetLastError();
  assert(status == cudaSuccess && "`MarlinGemm` CUDA kernel launch failed");
  if (c_tmp) {
    status = cudaFreeAsync(c_tmp, stream);
    assert(status == cudaSuccess &&
           "`MarlinGemm` failed to release the FP32 reduction buffer");
  }
  if (a_tmp) {
    status = cudaFreeAsync(a_tmp, stream);
    assert(status == cudaSuccess &&
           "`MarlinGemm` failed to release the act-order buffer");
  }
}

}  // namespace infini::ops
