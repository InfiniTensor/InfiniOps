#ifndef INFINI_OPS_BASE_ROTARY_EMBEDDING_H_
#define INFINI_OPS_BASE_ROTARY_EMBEDDING_H_

#include <cstddef>
#include <cstdint>
#include <optional>

#include "data_type.h"
#include "operator.h"

namespace infini::ops {

// vLLM-compatible rotary position embedding.
//
// Mirrors
// `vllm.model_executor.layers.rotary_embedding.RotaryEmbedding.forward`:
//   `forward(positions, query, key=None) -> (query, key | None)`.
//
// Inplace by default: passing `query_out = nullopt` / `key_out = nullopt`
// tells the kernel to write back into `query` / `key`, matching vLLM's
// inplace convention.  Callers that need a separate destination pass explicit
// out tensors.
//
// The previous `ApplyRotaryPosEmb` (pre-gathered fast path) is folded into
// this op via the `pre_gathered` constructor flag.  When
// `pre_gathered == true`, the caller has already executed
// `cos_sin_cache.index_select(0, positions)` plus any neox expansion; the
// kernel then skips the internal gather step.  vLLM's native contract uses
// `pre_gathered == false` (the default).
class RotaryEmbedding : public Operator<RotaryEmbedding> {
 public:
  // Parameter order mirrors vLLM's
  // `rotary_embedding(positions, query, key?, head_size, cos_sin_cache, is_neox)`
  // schema for the first 6 parameters.  Remaining parameters are InfiniOps
  // extensions:
  //   `rotary_dim`       — explicit; vLLM infers from
  //                        `cos_sin_cache.size(-1) / 2`.
  //   `query_out`        — optional out buffer (inplace when `nullopt`).
  //   `key_out`          — optional out buffer (inplace when `nullopt`).
  //   `pre_gathered`     — vendor-specific hint; `true` when the caller has
  //                        already run `cos_sin_cache.index_select(0, positions)`
  //                        plus neox expansion.
  //
  // Tensor conventions:
  //   `positions`        — `[T]`, `int64`.
  //   `query`            — `[T, Nq * head_size]` or `[T, Nq, head_size]`.
  //   `key`              — same layout as `query`; `nullopt` for MLA.
  //   `cos_sin_cache`    — default `[max_pos, rotary_dim * 2]` (cos then sin).
  //                        When `pre_gathered == true` the caller passes
  //                        `[T, head_size * 2]` already neox-expanded.
  //   `is_neox_style`    — `true` for NeoX split-half layout, `false` for
  //                        GPT-J interleaved.
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
           "`RotaryEmbedding`: `positions` must be `int64` (vLLM convention).");

    assert((query.ndim() == 2 || query.ndim() == 3) &&
           "`RotaryEmbedding`: `query` must be 2D `[T, Nq * head_size]` or 3D "
           "`[T, Nq, head_size]`.");

    if (key.has_value()) {
      assert((key->ndim() == 2 || key->ndim() == 3) &&
             "`RotaryEmbedding`: `key` must be 2D `[T, Nkv * head_size]` or "
             "3D `[T, Nkv, head_size]`.");
    }

    assert(rotary_dim <= head_size &&
           "`RotaryEmbedding`: `rotary_dim` must be `<= head_size`.");
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
