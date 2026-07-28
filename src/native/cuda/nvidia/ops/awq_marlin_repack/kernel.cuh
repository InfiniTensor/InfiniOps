// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from vLLM at commit 25ace8fe5df07fc13f4aef5a89db391f326e60ee:
// csrc/libtorch_stable/quantization/marlin/awq_marlin_repack.cu
// csrc/libtorch_stable/quantization/marlin/marlin.cuh

#ifndef INFINI_OPS_NVIDIA_AWQ_MARLIN_REPACK_KERNEL_CUH_
#define INFINI_OPS_NVIDIA_AWQ_MARLIN_REPACK_KERNEL_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infini::ops::awq_marlin_repack_detail {

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

template <int kNumBits, bool kIsA8Bit>
__global__ void AwqMarlinRepackKernel(
    const uint32_t* __restrict__ b_q_weight_ptr, uint32_t* __restrict__ out_ptr,
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
  constexpr int kTileNInts = kTargetTileNSize / kPackFactor;
  constexpr int kStageNThreads = kTileNInts / 4;
  constexpr int kStageKThreads = kTargetTileKSize;
  constexpr int kStageSize = kStageKThreads * kStageNThreads;

  const auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      CpAsyncFence();
      return;
    }

    const int first_n = n_tile_id * kTargetTileNSize;
    const int first_n_packed = first_n / kPackFactor;
    int4* shared_ptr = shared + kStageSize * pipe;

    if (threadIdx.x < kStageSize) {
      const auto k_id = threadIdx.x / kStageNThreads;
      const auto n_id = threadIdx.x % kStageNThreads;
      const int first_k = k_tile_id * kTargetTileKSize;

      CpAsync4(&shared_ptr[k_id * kStageNThreads + n_id],
               reinterpret_cast<const int4*>(
                   &b_q_weight_ptr[(first_k + k_id) * (size_n / kPackFactor) +
                                   first_n_packed + n_id * 4]));
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
    const int current_n_packed = current_n / kPackFactor;
    const int current_n_position = current_n % kPackFactor;

    constexpr int kSharedStride = kTileNInts;
    constexpr uint32_t kMask = (1U << kNumBits) - 1;
    int4* shared_stage_ptr = shared + kStageSize * pipe;
    auto* shared_stage_int_ptr = reinterpret_cast<uint32_t*>(shared_stage_ptr);

    int unpacked_n_position = 0;
    if constexpr (kNumBits == 4) {
      constexpr int kUndoPack[8] = {0, 4, 1, 5, 2, 6, 3, 7};
      unpacked_n_position = kUndoPack[current_n_position];
    } else {
      constexpr int kUndoPack[4] = {0, 2, 1, 3};
      unpacked_n_position = kUndoPack[current_n_position];
    }

    uint32_t values[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if constexpr (kIsA8Bit) {
        const int current_element = tensor_core_row + i;
        const uint32_t first_value =
            shared_stage_int_ptr[current_n_packed +
                                 (8 / kPackFactor) * (warp_id % 2) +
                                 kSharedStride * current_element];
        const uint32_t second_value =
            shared_stage_int_ptr[current_n_packed +
                                 (8 / kPackFactor) * (warp_id % 2) +
                                 kSharedStride * (current_element + 16)];

        values[i] = (first_value >> (unpacked_n_position * kNumBits)) & kMask;
        values[4 + i] =
            (second_value >> (unpacked_n_position * kNumBits)) & kMask;
      } else {
        const int current_element = tensor_core_row + kTensorCoreOffsets[i];
        const uint32_t first_value =
            shared_stage_int_ptr[current_n_packed +
                                 kSharedStride * current_element];
        const uint32_t second_value =
            shared_stage_int_ptr[current_n_packed + (8 / kPackFactor) +
                                 kSharedStride * current_element];

        values[i] = (first_value >> (unpacked_n_position * kNumBits)) & kMask;
        values[4 + i] =
            (second_value >> (unpacked_n_position * kNumBits)) & kMask;
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

}  // namespace infini::ops::awq_marlin_repack_detail

#endif  // INFINI_OPS_NVIDIA_AWQ_MARLIN_REPACK_KERNEL_CUH_
