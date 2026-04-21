#ifndef INFINI_OPS_BASE_ROPE_H_
#define INFINI_OPS_BASE_ROPE_H_

#include "operator.h"

namespace infini::ops {

// Applies rotary position embeddings (RoPE) to a single tensor `x`.
// Unlike the paired `RotaryEmbedding` operator, `Rope` takes
// `sin_cache` / `cos_cache` as separate inputs (not fused) and operates
// on a single tensor at a time, mirroring
// `infinicore::nn::RoPE::forward(x, pos)`.
//
// Expected shapes:
//   `x`:         `[..., head_dim]`.
//   `positions`: shape must be a prefix of `x.shape()[:-1]`; each entry
//                indexes into `sin_cache` / `cos_cache`.
//   `sin_cache`: `[max_seq_len, head_dim / 2]`.
//   `cos_cache`: `[max_seq_len, head_dim / 2]`.
//   `out`:       same shape as `x`; may alias `x` for in-place updates.
//
// `is_neox_style` selects between the two common RoPE pairing schemes:
//   - `true`  (GPT-NeoX / Llama): first half vs. second half of `head_dim`.
//   - `false` (GPT-J):            even vs. odd indices of `head_dim`.
class Rope : public Operator<Rope> {
 public:
  Rope(const Tensor x, const Tensor positions, const Tensor sin_cache,
       const Tensor cos_cache, bool is_neox_style, Tensor out)
      : x_type_{x.dtype()},
        positions_type_{positions.dtype()},
        sin_cache_type_{sin_cache.dtype()},
        cos_cache_type_{cos_cache.dtype()},
        out_type_{out.dtype()},
        x_shape_{x.shape()},
        positions_shape_{positions.shape()},
        sin_cache_shape_{sin_cache.shape()},
        cos_cache_shape_{cos_cache.shape()},
        out_shape_{out.shape()},
        x_strides_{x.strides()},
        positions_strides_{positions.strides()},
        sin_cache_strides_{sin_cache.strides()},
        cos_cache_strides_{cos_cache.strides()},
        out_strides_{out.strides()},
        is_neox_style_{is_neox_style} {
    assert(x.ndim() >= 2 && "`x` must have at least 2 dimensions");
    assert(sin_cache.ndim() == 2 && "`sin_cache` must be 2D");
    assert(cos_cache.ndim() == 2 && "`cos_cache` must be 2D");
    assert(x.shape().back() % 2 == 0 && "`head_dim` must be even");
    assert(sin_cache.shape()[1] == x.shape().back() / 2 &&
           "`sin_cache`'s last dim must equal `head_dim / 2`");
    assert(cos_cache.shape() == sin_cache.shape() &&
           "`sin_cache` and `cos_cache` must have the same shape");
    assert(positions.ndim() <= x.ndim() - 1 &&
           "`positions`' rank must not exceed `x.ndim() - 1`");
  }

  virtual void operator()(const Tensor x, const Tensor positions,
                          const Tensor sin_cache, const Tensor cos_cache,
                          bool is_neox_style, Tensor out) const = 0;

 protected:
  const DataType x_type_;

  const DataType positions_type_;

  const DataType sin_cache_type_;

  const DataType cos_cache_type_;

  const DataType out_type_;

  Tensor::Shape x_shape_;

  Tensor::Shape positions_shape_;

  Tensor::Shape sin_cache_shape_;

  Tensor::Shape cos_cache_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides x_strides_;

  Tensor::Strides positions_strides_;

  Tensor::Strides sin_cache_strides_;

  Tensor::Strides cos_cache_strides_;

  Tensor::Strides out_strides_;

  bool is_neox_style_{true};
};

}  // namespace infini::ops

#endif
