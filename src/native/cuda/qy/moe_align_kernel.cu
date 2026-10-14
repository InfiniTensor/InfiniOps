#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "native/cuda/qy/moe_align_kernel.h"

namespace infini::ops::qy {
namespace {

__global__ void moeAlignFillAndZero(int32_t *__restrict__ sorted_token_ids,
                                    int32_t *__restrict__ counts,
                                    int32_t *__restrict__ write_offsets,
                                    size_t sorted_capacity, size_t num_experts,
                                    size_t fill_value) {
  // Always seed sorted slots with numel so intra-expert padding is valid;
  // unused capacity beyond total is optionally cleared after scan.
  size_t n = sorted_capacity > num_experts ? sorted_capacity : num_experts;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (; idx < n; idx += stride) {
    if (idx < sorted_capacity) {
      sorted_token_ids[idx] = static_cast<int32_t>(fill_value);
    }
    if (idx < num_experts) {
      counts[idx] = 0;
      write_offsets[idx] = 0;
    }
  }
}

__global__ void moeAlignZeroTail(int32_t *__restrict__ sorted_token_ids,
                                 const int32_t *__restrict__ total_ptr,
                                 size_t sorted_capacity) {
  const size_t total = static_cast<size_t>(*total_ptr);
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = total + idx; i < sorted_capacity; i += stride) {
    sorted_token_ids[i] = 0;
  }
}

__global__ void moeAlignCount(const int32_t *__restrict__ topk_ids,
                              const int32_t *__restrict__ expert_map,
                              int32_t *__restrict__ counts, size_t numel,
                              size_t num_experts) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (; idx < numel; idx += stride) {
    int32_t expert_id = topk_ids[idx];
    if (expert_map != nullptr) {
      if (expert_id < 0 || expert_id >= static_cast<int32_t>(num_experts)) {
        continue;
      }
      expert_id = expert_map[expert_id];
    }
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(num_experts)) {
      continue;
    }
    atomicAdd(&counts[expert_id], 1);
  }
}

// Single-block exclusive scan of padded counts + expert_ids + total.
__global__ void moeAlignScan(const int32_t *__restrict__ counts,
                             int32_t *__restrict__ write_offsets,
                             int32_t *__restrict__ expert_ids,
                             int32_t *__restrict__ total_out,
                             int32_t num_experts, int32_t block_size,
                             int32_t expert_ids_capacity) {
  // Serial scan on thread 0 — E is modest (MiniCPM5 typically <= 256).
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int32_t cursor = 0;
  int32_t blocks_written = 0;
  for (int32_t e = 0; e < num_experts; ++e) {
    write_offsets[e] = cursor;
    int32_t c = counts[e];
    if (c <= 0) {
      continue;
    }
    int32_t padded = ((c + block_size - 1) / block_size) * block_size;
    int32_t n_blocks = padded / block_size;
    for (int32_t b = 0; b < n_blocks; ++b) {
      if (blocks_written < expert_ids_capacity) {
        expert_ids[blocks_written] = e;
      }
      ++blocks_written;
    }
    cursor += padded;
  }
  *total_out = cursor;
  // Zero unused expert_ids slots for determinism.
  for (int32_t i = blocks_written; i < expert_ids_capacity; ++i) {
    expert_ids[i] = -1;
  }
}

__global__ void moeAlignScatter(const int32_t *__restrict__ topk_ids,
                                const int32_t *__restrict__ expert_map,
                                int32_t *__restrict__ sorted_token_ids,
                                int32_t *__restrict__ write_offsets,
                                size_t numel, size_t num_experts,
                                size_t sorted_capacity) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (; idx < numel; idx += stride) {
    int32_t expert_id = topk_ids[idx];
    if (expert_map != nullptr) {
      if (expert_id < 0 || expert_id >= static_cast<int32_t>(num_experts)) {
        continue;
      }
      expert_id = expert_map[expert_id];
    }
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(num_experts)) {
      continue;
    }
    int32_t slot = atomicAdd(&write_offsets[expert_id], 1);
    if (slot >= 0 && static_cast<size_t>(slot) < sorted_capacity) {
      sorted_token_ids[slot] = static_cast<int32_t>(idx);
    }
  }
}

void checkLaunch(const char *name, cudaError_t err, size_t numel,
                 size_t num_experts, int grid, int block) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("QY moe_align_launch ") + name +
                             " failed: " + cudaGetErrorString(err) +
                             " (numel=" + std::to_string(numel) +
                             ", experts=" + std::to_string(num_experts) +
                             ", grid=" + std::to_string(grid) +
                             ", block=" + std::to_string(block) + ")");
  }
}

}  // namespace

void moe_align_launch(void *sorted_token_ids, void *expert_ids,
                      void *num_tokens_post_padded, const void *topk_ids,
                      const void *expert_map, size_t numel, size_t num_experts,
                      size_t block_size, size_t sorted_capacity,
                      size_t expert_ids_capacity, bool pad_sorted_token_ids,
                      void *workspace, void *stream) {
  if (workspace == nullptr) {
    throw std::runtime_error("QY moe_align_launch: workspace is null");
  }
  if (block_size == 0) {
    throw std::runtime_error("QY moe_align_launch: block_size must be > 0");
  }

  auto *counts = static_cast<int32_t *>(workspace);
  auto *write_offsets = counts + (num_experts + 1);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  constexpr int block = 128;
  auto grid_for = [](size_t n) {
    int g = static_cast<int>((n + block - 1) / block);
    return std::min(g, 128);
  };

  (void)cudaGetLastError();

  {
    size_t n = std::max(sorted_capacity, num_experts);
    int grid = grid_for(n == 0 ? 1 : n);
    moeAlignFillAndZero<<<grid, block, 0, cuda_stream>>>(
        static_cast<int32_t *>(sorted_token_ids), counts, write_offsets,
        sorted_capacity, num_experts, numel);
    checkLaunch("fill", cudaGetLastError(), numel, num_experts, grid, block);
  }

  if (numel > 0 && num_experts > 0) {
    int grid = grid_for(numel);
    moeAlignCount<<<grid, block, 0, cuda_stream>>>(
        static_cast<const int32_t *>(topk_ids),
        static_cast<const int32_t *>(expert_map), counts, numel, num_experts);
    checkLaunch("count", cudaGetLastError(), numel, num_experts, grid, block);
  }

  moeAlignScan<<<1, 1, 0, cuda_stream>>>(
      counts, write_offsets, static_cast<int32_t *>(expert_ids),
      static_cast<int32_t *>(num_tokens_post_padded),
      static_cast<int32_t>(num_experts), static_cast<int32_t>(block_size),
      static_cast<int32_t>(expert_ids_capacity));
  checkLaunch("scan", cudaGetLastError(), numel, num_experts, 1, 1);

  if (numel > 0 && num_experts > 0) {
    int grid = grid_for(numel);
    moeAlignScatter<<<grid, block, 0, cuda_stream>>>(
        static_cast<const int32_t *>(topk_ids),
        static_cast<const int32_t *>(expert_map),
        static_cast<int32_t *>(sorted_token_ids), write_offsets, numel,
        num_experts, sorted_capacity);
    checkLaunch("scatter", cudaGetLastError(), numel, num_experts, grid, block);
  }

  if (!pad_sorted_token_ids && sorted_capacity > 0) {
    // Match host path: unused capacity beyond total is zero, not numel.
    int grid = grid_for(sorted_capacity);
    moeAlignZeroTail<<<grid, block, 0, cuda_stream>>>(
        static_cast<int32_t *>(sorted_token_ids),
        static_cast<const int32_t *>(num_tokens_post_padded), sorted_capacity);
    checkLaunch("zero_tail", cudaGetLastError(), numel, num_experts, grid,
                block);
  }
}

}  // namespace infini::ops::qy
