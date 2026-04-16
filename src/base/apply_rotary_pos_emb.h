#ifndef INFINI_OPS_BASE_APPLY_ROTARY_POS_EMB_H_
#define INFINI_OPS_BASE_APPLY_ROTARY_POS_EMB_H_

#include <cstdint>

#include "operator.h"

namespace infini::ops {

// Apply rotary position embedding using pre-gathered cos/sin tensors.
//
// Unlike `RotaryEmbedding` which gathers cos/sin from a full
// `[max_seq_len, D]` cache using position indices, this operator takes
// pre-gathered `[T, D]` cos/sin directly.  This enables the caller to
// gather once per scheduling step and reuse across all model layers,
// eliminating redundant `IndexSelect` calls (e.g. 36 layers sharing the
// same positions in a single-batch LLM decode step).
//
// Accepts 2D `[T, N*D]` or 3D `[T, N, D]` query/key layouts.
// `num_heads_` and `num_kv_heads_` are derived from `numel / (T * D)`.
class ApplyRotaryPosEmb : public Operator<ApplyRotaryPosEmb> {
 public:
  // cos, sin: `[T, D]` pre-gathered, neox-expanded.
  // query: `[T, Nq*D]` or `[T, Nq, D]`.
  // key: `[T, Nkv*D]` or `[T, Nkv, D]`.
  ApplyRotaryPosEmb(const Tensor query, const Tensor key, const Tensor cos,
                    const Tensor sin, int64_t head_size, bool is_neox_style,
                    Tensor query_out, Tensor key_out)
      : num_tokens_{query.size(0)},
        num_heads_{static_cast<int64_t>(query.numel()) /
                   (static_cast<int64_t>(query.size(0)) * head_size)},
        num_kv_heads_{static_cast<int64_t>(key.numel()) /
                      (static_cast<int64_t>(key.size(0)) * head_size)},
        head_size_{head_size},
        is_neox_style_{is_neox_style} {
    assert((query.ndim() == 2 || query.ndim() == 3) &&
           "`ApplyRotaryPosEmb` requires query to be 2D or 3D");
    assert((key.ndim() == 2 || key.ndim() == 3) &&
           "`ApplyRotaryPosEmb` requires key to be 2D or 3D");
    assert(cos.ndim() == 2 && "`ApplyRotaryPosEmb` requires cos to be 2D "
                              "`[T, D]`");
    assert(sin.ndim() == 2 && "`ApplyRotaryPosEmb` requires sin to be 2D "
                              "`[T, D]`");
    assert(cos.size(0) == num_tokens_ &&
           "`ApplyRotaryPosEmb` requires cos.size(0) == T");
    assert(cos.size(1) == head_size &&
           "`ApplyRotaryPosEmb` requires cos.size(1) == head_size");
  }

  virtual void operator()(const Tensor query, const Tensor key,
                          const Tensor cos, const Tensor sin,
                          int64_t head_size, bool is_neox_style,
                          Tensor query_out, Tensor key_out) const = 0;

 protected:
  Tensor::Size num_tokens_{0};

  int64_t num_heads_{0};

  int64_t num_kv_heads_{0};

  int64_t head_size_{0};

  bool is_neox_style_{true};
};

}  // namespace infini::ops

#endif
