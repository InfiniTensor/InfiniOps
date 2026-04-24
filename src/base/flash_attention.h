#ifndef INFINI_OPS_BASE_FLASH_ATTENTION_H_
#define INFINI_OPS_BASE_FLASH_ATTENTION_H_

#include <cstddef>
#include <optional>
#include <vector>

#include "operator.h"

namespace infini::ops {

class FlashAttention : public Operator<FlashAttention> {
 public:
  // `window_left` / `window_right` is the native InfiniOps pair-form
  // window (left-context / right-context tokens, `-1` = disabled).
  // `sliding_window` is a vLLM-style single-parameter shortcut: when
  // set, it is normalized to `(sliding_window - 1, 0)` — i.e. causal
  // sliding over the most recent `sliding_window` tokens.  When both
  // forms are supplied the normalized values must agree.  Callers may
  // use whichever form is more natural; the kernel only sees the
  // resolved pair.
  FlashAttention(const Tensor query, const Tensor key, const Tensor value,
                 std::optional<Tensor> cu_seqlens_q,
                 std::optional<Tensor> cu_seqlens_kv,
                 std::optional<Tensor> block_table, int64_t num_heads,
                 int64_t num_kv_heads, int64_t head_size, double scale,
                 bool causal, int64_t window_left, int64_t window_right,
                 int64_t block_size, Tensor output,
                 std::optional<int64_t> sliding_window = std::nullopt)
      : num_tokens_{query.size(0)},
        num_heads_{num_heads},
        num_kv_heads_{num_kv_heads},
        head_size_{head_size},
        scale_{scale},
        causal_{causal},
        window_left_{resolveWindowLeft(window_left, sliding_window)},
        window_right_{resolveWindowRight(window_right, sliding_window)},
        block_size_{block_size},
        dtype_{query.dtype()},
        query_shape_{query.shape()},
        key_shape_{key.shape()},
        value_shape_{value.shape()},
        output_shape_{output.shape()},
        query_strides_{query.strides()},
        key_strides_{key.strides()},
        value_strides_{value.strides()},
        output_strides_{output.strides()},
        has_cu_seqlens_q_{cu_seqlens_q.has_value()},
        has_cu_seqlens_kv_{cu_seqlens_kv.has_value()},
        has_block_table_{block_table.has_value()} {
    assert(num_heads % num_kv_heads == 0 &&
           "`FlashAttention` requires `num_heads` divisible by `num_kv_heads`");
    assert(query.ndim() == 3 &&
           "`FlashAttention` requires query to be 3D [T, N, D]");
  }

  virtual void operator()(
      const Tensor query, const Tensor key, const Tensor value,
      std::optional<Tensor> cu_seqlens_q, std::optional<Tensor> cu_seqlens_kv,
      std::optional<Tensor> block_table, int64_t num_heads,
      int64_t num_kv_heads, int64_t head_size, double scale, bool causal,
      int64_t window_left, int64_t window_right, int64_t block_size,
      Tensor output,
      std::optional<int64_t> sliding_window = std::nullopt) const = 0;

 private:
  // Normalize the window representation.  If both the explicit pair and
  // `sliding_window` are supplied, assert the pair matches the derived
  // `(sliding_window - 1, 0)` causal-sliding window.
  static int64_t resolveWindowLeft(int64_t window_left,
                                   std::optional<int64_t> sliding_window) {
    if (!sliding_window.has_value()) return window_left;
    int64_t derived = sliding_window.value() - 1;
    assert(
        (window_left == -1 || window_left == derived) &&
        "`FlashAttention`: `window_left` inconsistent with `sliding_window`");
    return derived;
  }

  static int64_t resolveWindowRight(int64_t window_right,
                                    std::optional<int64_t> sliding_window) {
    if (!sliding_window.has_value()) return window_right;
    assert(
        (window_right == -1 || window_right == 0) &&
        "`FlashAttention`: `window_right` inconsistent with `sliding_window` "
        "(vLLM sliding_window implies right=0)");
    return 0;
  }

 public:
 protected:
  Tensor::Size num_tokens_{0};

  int64_t num_heads_{0};

  int64_t num_kv_heads_{0};

  int64_t head_size_{0};

  double scale_{0.0};

  bool causal_{false};

  int64_t window_left_{-1};

  int64_t window_right_{-1};

  int64_t block_size_{0};

  const DataType dtype_;

  Tensor::Shape query_shape_;

  Tensor::Shape key_shape_;

  Tensor::Shape value_shape_;

  Tensor::Shape output_shape_;

  Tensor::Strides query_strides_;

  Tensor::Strides key_strides_;

  Tensor::Strides value_strides_;

  Tensor::Strides output_strides_;

  bool has_cu_seqlens_q_{false};

  bool has_cu_seqlens_kv_{false};

  bool has_block_table_{false};
};

}  // namespace infini::ops

#endif
