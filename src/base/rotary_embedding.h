#ifndef INFINI_OPS_BASE_ROTARY_EMBEDDING_H_
#define INFINI_OPS_BASE_ROTARY_EMBEDDING_H_

#include <cstdint>
#include <optional>

#include "data_type.h"
#include "operator.h"

namespace infini::ops {

// Rotary position embedding.  First 6 parameters mirror vLLM's
// `rotary_embedding(positions, query, key?, head_size, cos_sin_cache,
// is_neox_style)` schema verbatim; `cos_sin_cache` is `[max_pos,
// rotary_dim * 2]` (cos then sin).  Inplace when `query_out` / `key_out`
// are `nullopt`.
class RotaryEmbedding : public Operator<RotaryEmbedding> {
 public:
  // `pre_gathered = true` means the caller has already applied
  // `cos_sin_cache.index_select(0, positions)` plus neox expansion, so
  // `cos_sin_cache` is laid out as `[T, head_size * 2]` and the kernel skips
  // the internal gather step.
  RotaryEmbedding(const Tensor positions, const Tensor query,
                  std::optional<Tensor> key, int64_t head_size,
                  const Tensor cos_sin_cache, bool is_neox_style,
                  int64_t rotary_dim,
                  std::optional<Tensor> query_out = std::nullopt,
                  std::optional<Tensor> key_out = std::nullopt,
                  bool pre_gathered = false)
      : num_tokens_{query.size(0)},
        num_heads_{static_cast<int64_t>(query.numel()) /
                   (static_cast<int64_t>(query.size(0)) * head_size)},
        num_kv_heads_{key.has_value()
                          ? static_cast<int64_t>(key->numel()) /
                                (static_cast<int64_t>(key->size(0)) * head_size)
                          : 0},
        head_size_{head_size},
        rotary_dim_{rotary_dim},
        is_neox_style_{is_neox_style},
        has_key_{key.has_value()},
        pre_gathered_{pre_gathered} {
    assert(positions.dtype() == DataType::kInt64 &&
           "`RotaryEmbedding`: `positions` must be `int64` (vLLM convention)");

    assert((query.ndim() == 2 || query.ndim() == 3) &&
           "`RotaryEmbedding`: `query` must be 2D `[T, Nq * head_size]` or 3D "
           "`[T, Nq, head_size]`");

    // TODO: relax once an MLA-capable Ascend impl lands.  The signature keeps
    // `std::optional<Tensor> key` for vLLM-API compatibility, but all current
    // Ascend impls assume `key` is present and rotate Q and K together.
    assert(key.has_value() &&
           "`RotaryEmbedding`: `key` is required; the `key = None` (MLA) path "
           "is not yet implemented on any backend");

    assert((key->ndim() == 2 || key->ndim() == 3) &&
           "`RotaryEmbedding`: `key` must be 2D `[T, Nkv * head_size]` or 3D "
           "`[T, Nkv, head_size]`");

    assert(rotary_dim <= head_size &&
           "`RotaryEmbedding`: `rotary_dim` must be `<= head_size`");
  }

  virtual void operator()(const Tensor positions, const Tensor query,
                          std::optional<Tensor> key, int64_t head_size,
                          const Tensor cos_sin_cache, bool is_neox_style,
                          int64_t rotary_dim,
                          std::optional<Tensor> query_out = std::nullopt,
                          std::optional<Tensor> key_out = std::nullopt,
                          bool pre_gathered = false) const = 0;

 protected:
  Tensor::Size num_tokens_{0};

  int64_t num_heads_{0};

  int64_t num_kv_heads_{0};

  int64_t head_size_{0};

  int64_t rotary_dim_{0};

  bool is_neox_style_{false};

  bool has_key_{false};

  bool pre_gathered_{false};
};

}  // namespace infini::ops

#endif
