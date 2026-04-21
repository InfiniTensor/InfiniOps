#ifndef INFINI_OPS_BASE_MHA_VARLEN_H_
#define INFINI_OPS_BASE_MHA_VARLEN_H_

#include "operator.h"

namespace infini::ops {

// Variable-length multi-head attention prefill over a paged KV cache.
//
// For each batch entry `b`, `q` contains `cum_seqlens_q[b+1] -
// cum_seqlens_q[b]` packed query tokens. The keys and values come from
// the first `cum_seqlens_k[b+1] - cum_seqlens_k[b]` positions of the
// paged cache selected by `block_table[b]`. The attention is causal
// with queries aligned to the end of the key range (i.e. query token
// `i` attends keys up to position `seqlen_k - seqlen_q + i`).
//
// Expected shapes:
//   `q`:             `[total_seqlen_q, num_heads, head_size]`.
//   `k_cache`:       `[num_blocks, block_size, num_kv_heads, head_size]`.
//   `v_cache`:       `[num_blocks, block_size, num_kv_heads, head_size]`.
//   `cum_seqlens_q`: `[batch_size + 1]` integer dtype.
//   `cum_seqlens_k`: `[batch_size + 1]` integer dtype.
//   `block_table`:   `[batch_size, max_num_blocks_per_seq]` integer dtype.
//   `out`:           `[total_seqlen_q, num_heads, head_size]`.
class MhaVarlen : public Operator<MhaVarlen> {
 public:
  MhaVarlen(const Tensor q, const Tensor k_cache, const Tensor v_cache,
            const Tensor cum_seqlens_q, const Tensor cum_seqlens_k,
            const Tensor block_table, float scale, Tensor out)
      : q_type_{q.dtype()},
        k_cache_type_{k_cache.dtype()},
        v_cache_type_{v_cache.dtype()},
        cum_seqlens_q_type_{cum_seqlens_q.dtype()},
        cum_seqlens_k_type_{cum_seqlens_k.dtype()},
        block_table_type_{block_table.dtype()},
        out_type_{out.dtype()},
        q_shape_{q.shape()},
        k_cache_shape_{k_cache.shape()},
        v_cache_shape_{v_cache.shape()},
        cum_seqlens_q_shape_{cum_seqlens_q.shape()},
        cum_seqlens_k_shape_{cum_seqlens_k.shape()},
        block_table_shape_{block_table.shape()},
        out_shape_{out.shape()},
        q_strides_{q.strides()},
        k_cache_strides_{k_cache.strides()},
        v_cache_strides_{v_cache.strides()},
        cum_seqlens_q_strides_{cum_seqlens_q.strides()},
        cum_seqlens_k_strides_{cum_seqlens_k.strides()},
        block_table_strides_{block_table.strides()},
        out_strides_{out.strides()},
        scale_{scale} {
    assert(q.ndim() == 3 && "`q` must be 3D `[total_seqlen_q, H_q, D]`");
    assert(k_cache.ndim() == 4 &&
           "`k_cache` must be 4D `[num_blocks, block_size, H_k, D]`");
    assert(v_cache.ndim() == 4 &&
           "`v_cache` must be 4D `[num_blocks, block_size, H_k, D]`");
    assert(cum_seqlens_q.ndim() == 1 && "`cum_seqlens_q` must be 1D");
    assert(cum_seqlens_k.ndim() == 1 && "`cum_seqlens_k` must be 1D");
    assert(block_table.ndim() == 2 && "`block_table` must be 2D");
    assert(out.ndim() == 3 && "`out` must be 3D `[total_seqlen_q, H_q, D]`");
    assert(q.dtype() == k_cache.dtype() && q.dtype() == v_cache.dtype() &&
           q.dtype() == out.dtype() &&
           "`q`, `k_cache`, `v_cache`, and `out` must share the same dtype");
  }

  virtual void operator()(const Tensor q, const Tensor k_cache,
                          const Tensor v_cache, const Tensor cum_seqlens_q,
                          const Tensor cum_seqlens_k,
                          const Tensor block_table, float scale,
                          Tensor out) const = 0;

 protected:
  const DataType q_type_;

  const DataType k_cache_type_;

  const DataType v_cache_type_;

  const DataType cum_seqlens_q_type_;

  const DataType cum_seqlens_k_type_;

  const DataType block_table_type_;

  const DataType out_type_;

  Tensor::Shape q_shape_;

  Tensor::Shape k_cache_shape_;

  Tensor::Shape v_cache_shape_;

  Tensor::Shape cum_seqlens_q_shape_;

  Tensor::Shape cum_seqlens_k_shape_;

  Tensor::Shape block_table_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides q_strides_;

  Tensor::Strides k_cache_strides_;

  Tensor::Strides v_cache_strides_;

  Tensor::Strides cum_seqlens_q_strides_;

  Tensor::Strides cum_seqlens_k_strides_;

  Tensor::Strides block_table_strides_;

  Tensor::Strides out_strides_;

  float scale_{1.0f};
};

}  // namespace infini::ops

#endif
