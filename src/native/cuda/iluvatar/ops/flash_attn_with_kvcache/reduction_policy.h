#ifndef INFINI_OPS_ILUVATAR_FLASH_DECODE_REDUCTION_POLICY_H_
#define INFINI_OPS_ILUVATAR_FLASH_DECODE_REDUCTION_POLICY_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace infini::ops::flash_attn_with_kvcache_iluvatar_detail {

struct DecodeQkConfig {
  bool all_bf16;
  bool int32_indices;
  bool contiguous_indices;
  bool has_alibi;
  std::size_t batch;
  std::size_t heads;
  std::size_t kv_heads;
  std::size_t head_size;
  std::size_t blocks_per_sequence;
  std::size_t page_size;
  std::ptrdiff_t q_batch_stride;
  std::ptrdiff_t q_head_stride;
  std::ptrdiff_t out_batch_stride;
  std::ptrdiff_t out_head_stride;
  std::ptrdiff_t k_block_stride;
  std::ptrdiff_t k_token_stride;
  std::ptrdiff_t k_head_stride;
  std::ptrdiff_t v_block_stride;
  std::ptrdiff_t v_token_stride;
  std::ptrdiff_t v_head_stride;
  std::uintptr_t q_address;
  std::uintptr_t k_address;
  std::uintptr_t v_address;
};

inline bool CanBatchDecodeQkReduction(const DecodeQkConfig& config) {
  constexpr std::size_t kMaxTokens =
      static_cast<std::size_t>(std::numeric_limits<int>::max()) - 8;
  if (!config.all_bf16 || !config.int32_indices || !config.contiguous_indices ||
      config.has_alibi || config.batch == 0 || config.batch > 16 ||
      config.heads == 0 || config.heads > 128 || config.kv_heads == 0 ||
      config.heads % config.kv_heads != 0 || config.head_size != 128 ||
      config.blocks_per_sequence == 0 || config.page_size == 0 ||
      config.page_size % 256 != 0 ||
      config.blocks_per_sequence > kMaxTokens / config.page_size) {
    return false;
  }
  const auto row_size = static_cast<std::ptrdiff_t>(config.heads * 128);
  if (config.q_head_stride != 128 || config.out_head_stride != 128 ||
      config.q_batch_stride < row_size || config.out_batch_stride < row_size ||
      config.q_batch_stride % 2 != 0 || config.q_address % 4 != 0 ||
      config.k_address % 16 != 0 || config.v_address % 16 != 0) {
    return false;
  }
  // The cache tile loads eight BF16 values per aligned vector.
  for (const auto stride :
       {config.k_block_stride, config.k_token_stride, config.k_head_stride,
        config.v_block_stride, config.v_token_stride, config.v_head_stride}) {
    if (stride <= 0 || stride % 8 != 0) return false;
  }
  return true;
}

inline int DecodeNumSplits(const DecodeQkConfig& config) {
  return config.blocks_per_sequence >= 32 && config.batch <= 4 ? 8 : 4;
}

}  // namespace infini::ops::flash_attn_with_kvcache_iluvatar_detail

#endif
