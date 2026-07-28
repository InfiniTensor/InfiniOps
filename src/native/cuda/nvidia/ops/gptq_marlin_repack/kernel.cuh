// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from vLLM at commit ffc4f08c8ee130d4ea6347c1bf31ffd4f8af28ab:
// csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu
// csrc/libtorch_stable/quantization/marlin/marlin.cuh

#ifndef INFINI_OPS_NVIDIA_GPTQ_MARLIN_REPACK_KERNEL_CUH_
#define INFINI_OPS_NVIDIA_GPTQ_MARLIN_REPACK_KERNEL_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infini::ops::gptq_marlin_repack_detail {

constexpr int kRepackStages = 8;
constexpr int kRepackThreads = 256;
constexpr int kTileSize = 16;
constexpr int kTileKSize = kTileSize;
constexpr int kTileNSize = kTileKSize * 4;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

__device__ inline void CpAsync4(void* shared_ptr, const void* global_ptr) {
  reinterpret_cast<int4*>(shared_ptr)[0] =
      reinterpret_cast<const int4*>(global_ptr)[0];
}

__device__ inline void CpAsyncFence() {}

template <int kCount>
__device__ inline void CpAsyncWait() {}

#else

__device__ inline void CpAsync4(void* shared_ptr, const void* global_ptr) {
  constexpr int kBytes = 16;
  const auto shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
               :
               : "r"(shared_address), "l"(global_ptr), "n"(kBytes));
}

__device__ inline void CpAsyncFence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int kCount>
__device__ inline void CpAsyncWait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(kCount));
}

#endif

template <int kNumBits, bool kHasPerm, bool kIsA8Bit>
__global__ void GptqMarlinRepackKernel(
    const uint32_t* __restrict__ b_q_weight_ptr,
    const uint32_t* __restrict__ perm_ptr, uint32_t* __restrict__ out_ptr,
    int size_k, int size_n) {
  constexpr int kPackFactor = 32 / kNumBits;
  constexpr int kTargetTileNSize = kTileNSize / (kIsA8Bit ? 2 : 1);
  constexpr int kTargetTileKSize = kTileKSize * (kIsA8Bit ? 2 : 1);
  const int k_tiles = size_k / kTargetTileKSize;
  const int n_tiles = size_n / kTargetTileNSize;
  const int block_k_tiles = (k_tiles + gridDim.x - 1) / gridDim.x;

  const auto start_k_tile = blockIdx.x * block_k_tiles;
  if (start_k_tile >= k_tiles) {
    return;
  }

  const int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

  const auto wait_for_stage = [&]() {
    CpAsyncWait<kRepackStages - 2>();
    __syncthreads();
  };

  extern __shared__ int4 shared[];
  constexpr int kPermSize = kTargetTileKSize / 4;
  int4* shared_perm_ptr = shared;
  int4* shared_pipe_ptr = shared_perm_ptr;
  if constexpr (kHasPerm) {
    shared_pipe_ptr += kPermSize;
  }

  constexpr int kTileInts = kTargetTileKSize / kPackFactor;
  constexpr int kStageNThreads = kTargetTileNSize / 4;
  constexpr int kStageKThreads = kHasPerm ? kTargetTileKSize : kTileInts;
  constexpr int kStageSize = kStageKThreads * kStageNThreads;

  const auto load_perm_to_shared = [&](int k_tile_id) {
    const int first_k_int4 = (k_tile_id * kTargetTileKSize) / 4;
    const auto* perm_int4_ptr = reinterpret_cast<const int4*>(perm_ptr);

    if (threadIdx.x < kPermSize) {
      shared_perm_ptr[threadIdx.x] = perm_int4_ptr[first_k_int4 + threadIdx.x];
    }
    __syncthreads();
  };

  const auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      CpAsyncFence();
      return;
    }

    const int first_n = n_tile_id * kTargetTileNSize;
    int4* shared_ptr = shared_pipe_ptr + kStageSize * pipe;

    if constexpr (kHasPerm) {
      if (threadIdx.x < kStageSize) {
        const auto k_id = threadIdx.x / kStageNThreads;
        const auto n_id = threadIdx.x % kStageNThreads;
        const auto* shared_perm_int_ptr =
            reinterpret_cast<const uint32_t*>(shared_perm_ptr);
        const int src_k = shared_perm_int_ptr[k_id];
        const int src_k_packed = src_k / kPackFactor;

        CpAsync4(
            &shared_ptr[k_id * kStageNThreads + n_id],
            reinterpret_cast<const int4*>(
                &b_q_weight_ptr[src_k_packed * size_n + first_n + n_id * 4]));
      }
    } else if (threadIdx.x < kStageSize) {
      const auto k_id = threadIdx.x / kStageNThreads;
      const auto n_id = threadIdx.x % kStageNThreads;
      const int first_k = k_tile_id * kTargetTileKSize;
      const int first_k_packed = first_k / kPackFactor;

      CpAsync4(&shared_ptr[k_id * kStageNThreads + n_id],
               reinterpret_cast<const int4*>(
                   &b_q_weight_ptr[(first_k_packed + k_id) * size_n + first_n +
                                   n_id * 4]));
    }

    CpAsyncFence();
  };

  const auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      return;
    }

    const auto warp_id = threadIdx.x / 32;
    const auto thread_id = threadIdx.x % 32;
    if (warp_id >= 4) {
      return;
    }

    const int tensor_core_column = thread_id / 4;
    const int tensor_core_row = (thread_id % 4) * (kIsA8Bit ? 4 : 2);
    constexpr int kTensorCoreOffsets[4] = {0, 1, 8, 9};
    const int current_n =
        (warp_id / (kIsA8Bit ? 2 : 1)) * 16 + tensor_core_column;

    constexpr int kSharedStride = kTargetTileNSize;
    constexpr uint32_t kMask = (1U << kNumBits) - 1;
    int4* shared_stage_ptr = shared_pipe_ptr + kStageSize * pipe;
    auto* shared_stage_int_ptr = reinterpret_cast<uint32_t*>(shared_stage_ptr);
    auto* shared_perm_int_ptr = reinterpret_cast<uint32_t*>(shared_perm_ptr);
    uint32_t values[8];

    if constexpr (kHasPerm) {
      static_assert(!kIsA8Bit);
      for (int i = 0; i < 4; ++i) {
        const int k_index = tensor_core_row + kTensorCoreOffsets[i];
        const uint32_t src_k = shared_perm_int_ptr[k_index];
        const uint32_t src_k_position = src_k % kPackFactor;
        const uint32_t first_value =
            shared_stage_int_ptr[k_index * kSharedStride + current_n];
        const uint32_t second_value =
            shared_stage_int_ptr[k_index * kSharedStride + current_n + 8];

        values[i] = (first_value >> (src_k_position * kNumBits)) & kMask;
        values[4 + i] = (second_value >> (src_k_position * kNumBits)) & kMask;
      }
    } else {
      uint32_t first_values[kTileInts];
      uint32_t second_values[kTileInts];

#pragma unroll
      for (int i = 0; i < kTileInts; ++i) {
        if constexpr (kIsA8Bit) {
          first_values[i] = shared_stage_int_ptr[current_n + kSharedStride * i +
                                                 (warp_id % 2) * 8];
        } else {
          first_values[i] = shared_stage_int_ptr[current_n + kSharedStride * i];
          second_values[i] =
              shared_stage_int_ptr[current_n + 8 + kSharedStride * i];
        }
      }

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int current_element =
            tensor_core_row + (kIsA8Bit ? i : kTensorCoreOffsets[i]);
        const int current_int = current_element / kPackFactor;
        const int current_position = current_element % kPackFactor;
        values[i] =
            (first_values[current_int] >> (current_position * kNumBits)) &
            kMask;
        if constexpr (kIsA8Bit) {
          values[4 + i] = (first_values[current_int + kTileInts / 2] >>
                           (current_position * kNumBits)) &
                          kMask;
        } else {
          values[4 + i] =
              (second_values[current_int] >> (current_position * kNumBits)) &
              kMask;
        }
      }
    }

    constexpr int kTileElements =
        kTargetTileKSize * kTargetTileNSize / kPackFactor;
    const int out_offset = (k_tile_id * n_tiles + n_tile_id) * kTileElements;

    // Matches FasterTransformer's interleaved numeric conversion layout:
    // https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
    if constexpr (!kIsA8Bit && kNumBits == 4) {
      constexpr int kPackIndices[8] = {0, 2, 4, 6, 1, 3, 5, 7};
      uint32_t result = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        result |= values[kPackIndices[i]] << (i * 4);
      }
      out_ptr[out_offset + thread_id * 4 + warp_id] = result;
    } else if constexpr (kIsA8Bit && kNumBits == 4) {
      constexpr int kPackIndices[8] = {0, 4, 1, 5, 2, 6, 3, 7};
      uint32_t result = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        result |= values[kPackIndices[i]] << (i * 4);
      }
      out_ptr[out_offset + thread_id * 4 + warp_id] = result;
    } else {
      constexpr int kPackIndices[4] = {0, 2, 1, 3};
      uint32_t first_result = 0;
      uint32_t second_result = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int index = kIsA8Bit ? i : kPackIndices[i];
        first_result |= values[index] << (i * 8);
        second_result |= values[4 + index] << (i * 8);
      }
      out_ptr[out_offset + thread_id * 8 + warp_id * 2] = first_result;
      out_ptr[out_offset + thread_id * 8 + warp_id * 2 + 1] = second_result;
    }
  };

  const auto start_pipes = [&](int k_tile_id, int n_tile_id) {
#pragma unroll
    for (int pipe = 0; pipe < kRepackStages - 1; ++pipe) {
      fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
    }
    wait_for_stage();
  };

#pragma unroll
  for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; ++k_tile_id) {
    int n_tile_id = 0;
    if constexpr (kHasPerm) {
      load_perm_to_shared(k_tile_id);
    }
    start_pipes(k_tile_id, n_tile_id);

    while (n_tile_id < n_tiles) {
#pragma unroll
      for (int pipe = 0; pipe < kRepackStages; ++pipe) {
        fetch_to_shared((pipe + kRepackStages - 1) % kRepackStages, k_tile_id,
                        n_tile_id + pipe + kRepackStages - 1);
        repack_tile(pipe, k_tile_id, n_tile_id + pipe);
        wait_for_stage();
      }
      n_tile_id += kRepackStages;
    }
  }
}

}  // namespace infini::ops::gptq_marlin_repack_detail

#endif  // INFINI_OPS_NVIDIA_GPTQ_MARLIN_REPACK_KERNEL_CUH_
