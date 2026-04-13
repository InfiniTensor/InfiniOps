#ifndef INFINI_OPS_BASE_ROTARY_EMBEDDING_H_
#define INFINI_OPS_BASE_ROTARY_EMBEDDING_H_

#include <cstddef>
#include <vector>

#include "operator.h"

namespace infini::ops {

// Rotary position embedding (RoPE) applied in-place to Q and K.
//
// Interface follows vLLM's `RotaryEmbedding.forward_oot()`:
//   `vllm.model_executor.layers.rotary_embedding.RotaryEmbedding`
//
// `positions`: `[T]` token position indices.
// `cos_sin_cache`: precomputed `[max_seq_len, rotary_dim]` table.
// `query` / `key`: `[T, N, D]` (TND layout), mutated in-place into
// `query_out` / `key_out`.
class RotaryEmbedding : public Operator<RotaryEmbedding> {
 public:
  RotaryEmbedding(const Tensor positions, const Tensor query, const Tensor key,
                  const Tensor cos_sin_cache, int64_t head_size,
                  int64_t rotary_dim, bool is_neox_style, Tensor query_out,
                  Tensor key_out)
      : num_tokens_{query.size(0)},
        num_heads_{static_cast<int64_t>(query.size(1))},
        num_kv_heads_{static_cast<int64_t>(key.size(1))},
        head_size_{head_size},
        rotary_dim_{rotary_dim},
        is_neox_style_{is_neox_style},
        query_shape_{query.shape()},
        key_shape_{key.shape()},
        cos_sin_cache_shape_{cos_sin_cache.shape()},
        query_out_shape_{query_out.shape()},
        key_out_shape_{key_out.shape()},
        query_strides_{query.strides()},
        key_strides_{key.strides()},
        query_out_strides_{query_out.strides()},
        key_out_strides_{key_out.strides()} {
    assert(query.ndim() == 3 &&
           "`RotaryEmbedding` requires query to be 3D [T, N, D]");
    assert(key.ndim() == 3 &&
           "`RotaryEmbedding` requires key to be 3D [T, N_kv, D]");
    assert(rotary_dim <= head_size &&
           "`RotaryEmbedding` requires rotary_dim <= head_size");
  }

  virtual void operator()(const Tensor positions, const Tensor query,
                          const Tensor key, const Tensor cos_sin_cache,
                          int64_t head_size, int64_t rotary_dim,
                          bool is_neox_style, Tensor query_out,
                          Tensor key_out) const = 0;

 protected:
  Tensor::Size num_tokens_{0};

  int64_t num_heads_{0};

  int64_t num_kv_heads_{0};

  int64_t head_size_{0};

  int64_t rotary_dim_{0};

  bool is_neox_style_{true};

  Tensor::Shape query_shape_;

  Tensor::Shape key_shape_;

  Tensor::Shape cos_sin_cache_shape_;

  Tensor::Shape query_out_shape_;

  Tensor::Shape key_out_shape_;

  Tensor::Strides query_strides_;

  Tensor::Strides key_strides_;

  Tensor::Strides query_out_strides_;

  Tensor::Strides key_out_strides_;
};

}  // namespace infini::ops

#endif
