#ifndef INFINI_OPS_BASE_PAGED_ATTENTION_H_
#define INFINI_OPS_BASE_PAGED_ATTENTION_H_

#include <cstddef>
#include <cstdint>
#include <optional>

#include "operator.h"

namespace infini::ops {

// Paged decode attention operator.
//
// Performs multi-head attention over paged KV caches for decode (single-token
// queries per sequence).
//
// Interface follows vLLM's paged attention convention:
//   - vLLM CUDA: `torch.ops.vllm.paged_attention_v1` uses the same query
//     shape [batch, num_heads, head_size] and seq_lens [batch] int32.
//     KV cache differs (5D on CUDA for vectorization, 4D here).
//   - vLLM-Ascend: `torch_npu._npu_paged_attention` wraps ATB
//     `PagedAttentionParam` with default `inputLayout` (`TYPE_BSND`).
//   - ATB `PagedAttentionParam`: `headNum`, `kvHeadNum`, `qkScale`,
//     `maskType` (default NORM), `inputLayout` (default `TYPE_BSND`).
//
// Input layout (BSND with S=1 for decode):
//   query       : [batch, num_heads, head_size]
//   key_cache   : [num_blocks, block_size, num_kv_heads, head_size]
//   value_cache : [num_blocks, block_size, num_kv_heads, head_size]
//   seq_lens    : [batch] int32 — total context length per sequence
//   block_table : [batch, max_num_blocks_per_seq] int32
//
// Output layout:
//   output      : [batch, num_heads, head_size]
class PagedAttention : public Operator<PagedAttention> {
 public:
  PagedAttention(const Tensor query, const Tensor key_cache,
                 const Tensor value_cache, const Tensor seq_lens,
                 const Tensor block_table, int64_t num_heads,
                 int64_t num_kv_heads, int64_t head_size, double scale,
                 int64_t block_size, Tensor output,
                 std::optional<Tensor> seq_lens_host = std::nullopt,
                 std::optional<Tensor> block_table_host = std::nullopt)
      : batch_size_{query.size(0)},
        num_heads_{num_heads},
        num_kv_heads_{num_kv_heads},
        head_size_{head_size},
        scale_{scale},
        block_size_{block_size},
        dtype_{query.dtype()},
        query_shape_{query.shape()},
        key_cache_shape_{key_cache.shape()},
        value_cache_shape_{value_cache.shape()},
        seq_lens_shape_{seq_lens.shape()},
        block_table_shape_{block_table.shape()},
        output_shape_{output.shape()},
        has_seq_lens_host_{seq_lens_host.has_value()},
        has_block_table_host_{block_table_host.has_value()} {
    assert(
        num_heads % num_kv_heads == 0 &&
        "`PagedAttention` requires `num_heads` divisible by `num_kv_heads`.");
    assert(query.ndim() == 3 &&
           "`PagedAttention` requires query to be 3D [batch, num_heads, "
           "head_size].");
    assert(key_cache.ndim() == 4 &&
           "`PagedAttention` requires key_cache to be 4D [num_blocks, "
           "block_size, num_kv_heads, head_size].");
    assert(seq_lens.ndim() == 1 &&
           "`PagedAttention` requires seq_lens to be 1D [batch].");
    assert(block_table.ndim() == 2 &&
           "`PagedAttention` requires block_table to be 2D [batch, "
           "max_num_blocks].");
  }

  virtual void operator()(
      const Tensor query, const Tensor key_cache, const Tensor value_cache,
      const Tensor seq_lens, const Tensor block_table, int64_t num_heads,
      int64_t num_kv_heads, int64_t head_size, double scale, int64_t block_size,
      Tensor output, std::optional<Tensor> seq_lens_host = std::nullopt,
      std::optional<Tensor> block_table_host = std::nullopt) const = 0;

 protected:
  Tensor::Size batch_size_{0};

  int64_t num_heads_{0};

  int64_t num_kv_heads_{0};

  int64_t head_size_{0};

  double scale_{0.0};

  int64_t block_size_{0};

  const DataType dtype_;

  Tensor::Shape query_shape_;

  Tensor::Shape key_cache_shape_;

  Tensor::Shape value_cache_shape_;

  Tensor::Shape seq_lens_shape_;

  Tensor::Shape block_table_shape_;

  Tensor::Shape output_shape_;

  bool has_seq_lens_host_{false};

  bool has_block_table_host_{false};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_PAGED_ATTENTION_H_
