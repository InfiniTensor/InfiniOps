#pragma once

#include <cuda_runtime.h>

#include "data_type.h"
#include "native/cuda/nvidia/data_type_.h"

namespace infini::ops::native_flash_mla {

struct MlaPrefillLaunch {
  std::size_t num_heads, num_seqs, num_kv_heads, total_qtokens;
  std::size_t max_num_blocks_per_seq, block_size;
  float scale;
  std::ptrdiff_t block_table_batch_stride, q_stride, q_head_stride;
  std::ptrdiff_t k_batch_stride, k_row_stride, k_head_stride;
  std::ptrdiff_t v_batch_stride, v_row_stride, v_head_stride;
  std::ptrdiff_t o_stride, o_head_stride;
};

struct MlaDecodeLaunch {
  std::size_t num_heads, num_seqs, num_kv_heads;
  std::size_t max_num_blocks_per_seq, block_size;
  float scale;
  std::ptrdiff_t block_table_batch_stride;
  std::ptrdiff_t q_batch_stride, q_head_stride;
  std::ptrdiff_t k_block_stride, k_head_stride, k_slot_stride;
  std::ptrdiff_t v_block_stride, v_head_stride, v_slot_stride;
  std::ptrdiff_t o_batch_stride, o_head_stride;
  std::ptrdiff_t seqlens_stride;
};

void launch_mla_prefill(void* out, const void* q, const void* k_cache,
                        const void* v_cache, const void* block_tables,
                        const void* cu_seqlens_k, const void* cu_seqlens_q,
                        DataType data_dtype, DataType index_dtype,
                        const MlaPrefillLaunch& p, cudaStream_t stream);

void launch_mla_decode(void* out, const void* q, const void* k_cache,
                       const void* v_cache, const void* block_tables,
                       const void* seqlens, DataType data_dtype,
                       DataType index_dtype, const MlaDecodeLaunch& p,
                       cudaStream_t stream);

}  // namespace infini::ops::native_flash_mla
