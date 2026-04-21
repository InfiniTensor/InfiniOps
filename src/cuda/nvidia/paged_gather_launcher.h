#ifndef INFINI_OPS_NVIDIA_PAGED_GATHER_LAUNCHER_H_
#define INFINI_OPS_NVIDIA_PAGED_GATHER_LAUNCHER_H_

#include <cstddef>
#include <cstdint>

#include "data_type.h"

namespace infini::ops {

// C++-visible launcher for the single-sequence paged-gather CUDA kernel.
// Defined in `paged_gather_launcher.cu` (compiled by `nvcc`). Intended to
// be called from host-side code (e.g. the PyTorch backend) to replace an
// `arange + div_floor + remainder + gather + advanced-index` ATen chain
// with a single kernel launch.
//
// Requires `dense_k` and `dense_v` pre-allocated with shape `[seqlen_k,
// num_kv_heads, head_size]` and the same `dtype` as the cache.
// `block_table` points at the row for the sequence being gathered and has
// `ceil(seqlen_k / block_size)` valid entries; its `dtype` must be
// `kInt32` or `kInt64`.
void LaunchPagedGatherNvidia(
    void *dense_k, void *dense_v,
    const void *k_cache, const void *v_cache,
    const void *block_table, DataType cache_dtype,
    DataType block_table_dtype, std::size_t seqlen_k,
    std::size_t num_kv_heads, std::size_t head_size, std::size_t block_size,
    std::ptrdiff_t k_cache_block_stride, std::ptrdiff_t v_cache_block_stride,
    std::ptrdiff_t k_cache_slot_stride, std::ptrdiff_t v_cache_slot_stride,
    std::ptrdiff_t k_cache_head_stride, std::ptrdiff_t v_cache_head_stride,
    std::ptrdiff_t dense_seqlen_stride, std::ptrdiff_t dense_head_stride,
    void *stream);

}  // namespace infini::ops

#endif
