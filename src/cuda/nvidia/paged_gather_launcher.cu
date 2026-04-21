#include "cuda/nvidia/paged_gather_launcher.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

#include "cuda/nvidia/runtime_.h"
#include "cuda/paged_gather/kernel.cuh"

namespace infini::ops {

namespace {

template <typename TData, typename TBlockIdx, int BLOCK_SIZE>
void launch_typed(void *dense_k, void *dense_v, const void *k_cache,
                  const void *v_cache, const void *block_table,
                  std::size_t seqlen_k, std::size_t num_kv_heads,
                  std::size_t head_size, std::size_t block_size,
                  std::ptrdiff_t k_cache_block_stride,
                  std::ptrdiff_t v_cache_block_stride,
                  std::ptrdiff_t k_cache_slot_stride,
                  std::ptrdiff_t v_cache_slot_stride,
                  std::ptrdiff_t k_cache_head_stride,
                  std::ptrdiff_t v_cache_head_stride,
                  std::ptrdiff_t dense_seqlen_stride,
                  std::ptrdiff_t dense_head_stride, cudaStream_t stream) {
  dim3 grid(static_cast<unsigned int>(num_kv_heads),
            static_cast<unsigned int>(seqlen_k));
  PagedGatherKernel<TData, TBlockIdx, BLOCK_SIZE>
      <<<grid, BLOCK_SIZE, 0, stream>>>(
          reinterpret_cast<TData *>(dense_k),
          reinterpret_cast<TData *>(dense_v),
          reinterpret_cast<const TData *>(k_cache),
          reinterpret_cast<const TData *>(v_cache),
          reinterpret_cast<const TBlockIdx *>(block_table), block_size,
          head_size, k_cache_block_stride, v_cache_block_stride,
          k_cache_slot_stride, v_cache_slot_stride, k_cache_head_stride,
          v_cache_head_stride, dense_seqlen_stride, dense_head_stride);
}

template <typename TData, int BLOCK_SIZE>
void dispatch_block_table(DataType block_table_dtype, void *dense_k,
                          void *dense_v, const void *k_cache,
                          const void *v_cache, const void *block_table,
                          std::size_t seqlen_k, std::size_t num_kv_heads,
                          std::size_t head_size, std::size_t block_size,
                          std::ptrdiff_t k_cache_block_stride,
                          std::ptrdiff_t v_cache_block_stride,
                          std::ptrdiff_t k_cache_slot_stride,
                          std::ptrdiff_t v_cache_slot_stride,
                          std::ptrdiff_t k_cache_head_stride,
                          std::ptrdiff_t v_cache_head_stride,
                          std::ptrdiff_t dense_seqlen_stride,
                          std::ptrdiff_t dense_head_stride,
                          cudaStream_t stream) {
  if (block_table_dtype == DataType::kInt32) {
    launch_typed<TData, std::int32_t, BLOCK_SIZE>(
        dense_k, dense_v, k_cache, v_cache, block_table, seqlen_k,
        num_kv_heads, head_size, block_size, k_cache_block_stride,
        v_cache_block_stride, k_cache_slot_stride, v_cache_slot_stride,
        k_cache_head_stride, v_cache_head_stride, dense_seqlen_stride,
        dense_head_stride, stream);
  } else if (block_table_dtype == DataType::kInt64) {
    launch_typed<TData, std::int64_t, BLOCK_SIZE>(
        dense_k, dense_v, k_cache, v_cache, block_table, seqlen_k,
        num_kv_heads, head_size, block_size, k_cache_block_stride,
        v_cache_block_stride, k_cache_slot_stride, v_cache_slot_stride,
        k_cache_head_stride, v_cache_head_stride, dense_seqlen_stride,
        dense_head_stride, stream);
  } else {
    throw std::invalid_argument(
        "`LaunchPagedGatherNvidia`: `block_table` must be `int32` or `int64`");
  }
}

}  // namespace

void LaunchPagedGatherNvidia(
    void *dense_k, void *dense_v, const void *k_cache, const void *v_cache,
    const void *block_table, DataType cache_dtype, DataType block_table_dtype,
    std::size_t seqlen_k, std::size_t num_kv_heads, std::size_t head_size,
    std::size_t block_size, std::ptrdiff_t k_cache_block_stride,
    std::ptrdiff_t v_cache_block_stride, std::ptrdiff_t k_cache_slot_stride,
    std::ptrdiff_t v_cache_slot_stride, std::ptrdiff_t k_cache_head_stride,
    std::ptrdiff_t v_cache_head_stride, std::ptrdiff_t dense_seqlen_stride,
    std::ptrdiff_t dense_head_stride, void *stream) {
  if (seqlen_k == 0) {
    return;
  }

  auto cuda_stream = static_cast<cudaStream_t>(stream);
  constexpr int kBlockSize = 128;

  switch (cache_dtype) {
    case DataType::kFloat16:
      dispatch_block_table<__half, kBlockSize>(
          block_table_dtype, dense_k, dense_v, k_cache, v_cache, block_table,
          seqlen_k, num_kv_heads, head_size, block_size, k_cache_block_stride,
          v_cache_block_stride, k_cache_slot_stride, v_cache_slot_stride,
          k_cache_head_stride, v_cache_head_stride, dense_seqlen_stride,
          dense_head_stride, cuda_stream);
      break;
    case DataType::kBFloat16:
      dispatch_block_table<__nv_bfloat16, kBlockSize>(
          block_table_dtype, dense_k, dense_v, k_cache, v_cache, block_table,
          seqlen_k, num_kv_heads, head_size, block_size, k_cache_block_stride,
          v_cache_block_stride, k_cache_slot_stride, v_cache_slot_stride,
          k_cache_head_stride, v_cache_head_stride, dense_seqlen_stride,
          dense_head_stride, cuda_stream);
      break;
    case DataType::kFloat32:
      dispatch_block_table<float, kBlockSize>(
          block_table_dtype, dense_k, dense_v, k_cache, v_cache, block_table,
          seqlen_k, num_kv_heads, head_size, block_size, k_cache_block_stride,
          v_cache_block_stride, k_cache_slot_stride, v_cache_slot_stride,
          k_cache_head_stride, v_cache_head_stride, dense_seqlen_stride,
          dense_head_stride, cuda_stream);
      break;
    default:
      throw std::invalid_argument(
          "`LaunchPagedGatherNvidia`: unsupported `cache_dtype`");
  }
}

}  // namespace infini::ops
