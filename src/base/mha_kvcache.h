#ifndef INFINI_OPS_BASE_MHA_KVCACHE_H_
#define INFINI_OPS_BASE_MHA_KVCACHE_H_

#include "operator.h"

namespace infini::ops {

// Multi-head attention decode over a paged KV cache.
//
// Each sequence in the batch attends its query tokens (typically one per
// sequence during decode) to the first `seqlens_k[b]` key tokens, which
// `block_table[b]` points to inside the paged `(k_cache, v_cache)`.
//
// Expected shapes:
//   `q`:           `[batch_size, seqlen_q, num_heads, head_size]`.
//   `k_cache`:     `[num_blocks, block_size, num_kv_heads, head_size]`.
//   `v_cache`:     `[num_blocks, block_size, num_kv_heads, head_size]`.
//   `seqlens_k`:   `[batch_size]` integer dtype.
//   `block_table`: `[batch_size, max_num_blocks_per_seq]` integer dtype.
//   `out`:         `[batch_size, seqlen_q, num_heads, head_size]`.
class MhaKvcache : public Operator<MhaKvcache> {
 public:
  MhaKvcache(const Tensor q, const Tensor k_cache, const Tensor v_cache,
             const Tensor seqlens_k, const Tensor block_table, float scale,
             Tensor out)
      : q_type_{q.dtype()},
        k_cache_type_{k_cache.dtype()},
        v_cache_type_{v_cache.dtype()},
        seqlens_k_type_{seqlens_k.dtype()},
        block_table_type_{block_table.dtype()},
        out_type_{out.dtype()},
        q_shape_{q.shape()},
        k_cache_shape_{k_cache.shape()},
        v_cache_shape_{v_cache.shape()},
        seqlens_k_shape_{seqlens_k.shape()},
        block_table_shape_{block_table.shape()},
        out_shape_{out.shape()},
        q_strides_{q.strides()},
        k_cache_strides_{k_cache.strides()},
        v_cache_strides_{v_cache.strides()},
        seqlens_k_strides_{seqlens_k.strides()},
        block_table_strides_{block_table.strides()},
        out_strides_{out.strides()},
        scale_{scale} {
    assert(q.ndim() == 4 && "`q` must be 4D `[B, S_q, H_q, D]`");
    assert(k_cache.ndim() == 4 &&
           "`k_cache` must be 4D `[num_blocks, block_size, H_k, D]`");
    assert(v_cache.ndim() == 4 && "`v_cache` must be 4D `[num_blocks, block_size, H_k, D]`");
    assert(seqlens_k.ndim() == 1 && "`seqlens_k` must be 1D");
    assert(block_table.ndim() == 2 && "`block_table` must be 2D");
    assert(out.ndim() == 4 && "`out` must be 4D `[B, S_q, H_q, D]`");
    assert(q.dtype() == k_cache.dtype() && q.dtype() == v_cache.dtype() &&
           q.dtype() == out.dtype() &&
           "`q`, `k_cache`, `v_cache`, and `out` must share the same dtype");
  }

  virtual void operator()(const Tensor q, const Tensor k_cache,
                          const Tensor v_cache, const Tensor seqlens_k,
                          const Tensor block_table, float scale,
                          Tensor out) const = 0;

 protected:
  const DataType q_type_;

  const DataType k_cache_type_;

  const DataType v_cache_type_;

  const DataType seqlens_k_type_;

  const DataType block_table_type_;

  const DataType out_type_;

  Tensor::Shape q_shape_;

  Tensor::Shape k_cache_shape_;

  Tensor::Shape v_cache_shape_;

  Tensor::Shape seqlens_k_shape_;

  Tensor::Shape block_table_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides q_strides_;

  Tensor::Strides k_cache_strides_;

  Tensor::Strides v_cache_strides_;

  Tensor::Strides seqlens_k_strides_;

  Tensor::Strides block_table_strides_;

  Tensor::Strides out_strides_;

  float scale_{1.0f};
};

}  // namespace infini::ops

#endif
