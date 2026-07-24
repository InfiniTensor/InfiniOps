#ifndef INFINI_OPS_BASE_MHA_FWD_KVCACHE_H_
#define INFINI_OPS_BASE_MHA_FWD_KVCACHE_H_

#include <cassert>
#include <cmath>
#include <optional>

#include "data_type.h"
#include "operator.h"

namespace infini::ops {

// FlashAttention-compatible forward attention over an existing KV cache.
//
// Layout follows `flash_attn` `fwd_kvcache`:
//   `q`: `[batch, seqlen_q, num_heads, head_size]`.
//   `kcache` / `vcache`: `[batch_cache, seqlen_k, num_kv_heads, head_size]`
//     or paged `[num_blocks, page_block_size, num_kv_heads, head_size]` when
//     `block_table` is supplied.
//
// InfiniOps is an in-place operator API, so `out` must be supplied even
// though FlashAttention can allocate it when omitted.
// This signature intentionally keeps `out` near the FlashAttention argument
// position for compatibility with existing callers; this is an exception to
// the normal InfiniOps output-last convention.
class MhaFwdKvcache : public Operator<MhaFwdKvcache> {
 public:
  MhaFwdKvcache(
      const Tensor q, const Tensor kcache, const Tensor vcache,
      std::optional<Tensor> k, std::optional<Tensor> v,
      std::optional<Tensor> seqlens_k, std::optional<Tensor> rotary_cos,
      std::optional<Tensor> rotary_sin, std::optional<Tensor> cache_batch_idx,
      std::optional<Tensor> leftpad_k, std::optional<Tensor> block_table,
      std::optional<Tensor> alibi_slopes, Tensor out, float softmax_scale,
      bool is_causal, int64_t window_size_left, int64_t window_size_right,
      float softcap, bool is_rotary_interleaved, int64_t num_splits = 0)
      : batch_size_{q.size(0)},
        seqlen_q_{q.size(1)},
        num_heads_{q.size(2)},
        head_size_{q.size(3)},
        num_kv_heads_{kcache.size(2)},
        page_block_size_{block_table.has_value() ? kcache.size(1) : 0},
        softmax_scale_{softmax_scale},
        is_causal_{is_causal},
        window_size_left_{window_size_left},
        window_size_right_{window_size_right},
        softcap_{softcap},
        is_rotary_interleaved_{is_rotary_interleaved},
        num_splits_{num_splits},
        q_dtype_{q.dtype()},
        has_k_{k.has_value()},
        has_v_{v.has_value()},
        has_seqlens_k_{seqlens_k.has_value()},
        has_rotary_cos_{rotary_cos.has_value()},
        has_rotary_sin_{rotary_sin.has_value()},
        has_cache_batch_idx_{cache_batch_idx.has_value()},
        has_leftpad_k_{leftpad_k.has_value()},
        has_block_table_{block_table.has_value()},
        has_alibi_slopes_{alibi_slopes.has_value()} {
    assert(q.ndim() == 4 &&
           "`MhaFwdKvcache` requires `q` to be `[batch, seq, heads, dim]`");
    assert(kcache.ndim() == 4 && vcache.ndim() == 4 &&
           "`MhaFwdKvcache` requires `kcache` / `vcache` to be 4D.");
    assert(kcache.shape() == vcache.shape() &&
           "`MhaFwdKvcache` requires `kcache` and `vcache` same shape");
    assert(kcache.dtype() == q.dtype() && vcache.dtype() == q.dtype() &&
           "`MhaFwdKvcache` requires `q`, `kcache`, and `vcache` same dtype");
    assert(out.dtype() == q.dtype() &&
           "`MhaFwdKvcache` requires `out` to have same dtype as `q`.");
    assert(kcache.size(3) == head_size_ &&
           "`MhaFwdKvcache` requires cache head dim to match `q`.");
    assert(q.stride(-1) == 1 && kcache.stride(-1) == 1 &&
           vcache.stride(-1) == 1 &&
           "`MhaFwdKvcache` requires contiguous last dimension");
    assert(num_heads_ % num_kv_heads_ == 0 &&
           "`MhaFwdKvcache` requires `num_heads` divisible by `num_kv_heads`");
    assert(head_size_ <= 256 &&
           "`MhaFwdKvcache` supports `head_size` up to 256");
    assert(out.shape() == q.shape() &&
           "`MhaFwdKvcache` requires `out` to match `q` shape");
    assert(out.stride(-1) == 1 &&
           "`MhaFwdKvcache` requires `out` contiguous last dimension");
    assert(std::isfinite(softmax_scale_) &&
           "`MhaFwdKvcache` requires finite `softmax_scale`.");

    if (k.has_value() || v.has_value()) {
      assert(k.has_value() && v.has_value() &&
             "`MhaFwdKvcache` requires `k` and `v` together.");
      if (k.has_value() && v.has_value()) {
        assert(k->ndim() == 4 && v->ndim() == 4 &&
               "`MhaFwdKvcache` requires appended `k` / `v` to be 4D.");
        assert(k->shape() == v->shape() &&
               "`MhaFwdKvcache` requires appended `k` and `v` same shape.");
        assert(k->dtype() == q.dtype() && v->dtype() == q.dtype() &&
               "`MhaFwdKvcache` requires appended `k` / `v` same dtype as "
               "`q`.");
        assert(k->size(0) == batch_size_ && k->size(2) == num_kv_heads_ &&
               k->size(3) == head_size_ &&
               "`MhaFwdKvcache` requires appended `k` / `v` to match cache "
               "batch, heads, and dim.");
        assert(k->stride(-1) == 1 && v->stride(-1) == 1 &&
               "`MhaFwdKvcache` requires appended `k` / `v` contiguous last "
               "dimension.");
      }
    }

    if (seqlens_k.has_value()) {
      assert(seqlens_k->ndim() == 1 &&
             "`MhaFwdKvcache` requires `seqlens_k` to be 1D.");
      assert((seqlens_k->dtype() == DataType::kInt32 ||
              seqlens_k->dtype() == DataType::kInt64) &&
             "`MhaFwdKvcache` requires `seqlens_k` to be `int32` or `int64`.");
    }

    if (block_table.has_value()) {
      assert(block_table->ndim() == 2 &&
             "`MhaFwdKvcache` requires `block_table` to be 2D.");
      assert(block_table->dtype() == DataType::kInt32 &&
             "`MhaFwdKvcache` requires `block_table` to be `int32`.");
    }

    if (cache_batch_idx.has_value()) {
      assert(cache_batch_idx->ndim() == 1 &&
             "`MhaFwdKvcache` requires `cache_batch_idx` to be 1D.");
      assert(cache_batch_idx->dtype() == DataType::kInt32 &&
             "`MhaFwdKvcache` requires `cache_batch_idx` to be `int32`.");
    }

    if (leftpad_k.has_value()) {
      assert(leftpad_k->ndim() == 1 &&
             "`MhaFwdKvcache` requires `leftpad_k` to be 1D.");
      assert(leftpad_k->dtype() == DataType::kInt32 &&
             "`MhaFwdKvcache` requires `leftpad_k` to be `int32`.");
    }

    if (rotary_cos.has_value() || rotary_sin.has_value()) {
      assert(rotary_cos.has_value() && rotary_sin.has_value() &&
             "`MhaFwdKvcache` requires `rotary_cos` and `rotary_sin` "
             "together.");
      if (rotary_cos.has_value() && rotary_sin.has_value()) {
        assert(rotary_cos->shape() == rotary_sin->shape() &&
               "`MhaFwdKvcache` requires rotary tensors to have same shape.");
      }
    }

    if (alibi_slopes.has_value()) {
      assert((alibi_slopes->ndim() == 1 || alibi_slopes->ndim() == 2) &&
             "`MhaFwdKvcache` requires `alibi_slopes` to be 1D or 2D.");
      assert(alibi_slopes->dtype() == DataType::kFloat32 &&
             "`MhaFwdKvcache` requires `alibi_slopes` to be `float32`.");
    }
  }

  virtual void operator()(
      const Tensor q, const Tensor kcache, const Tensor vcache,
      std::optional<Tensor> k, std::optional<Tensor> v,
      std::optional<Tensor> seqlens_k, std::optional<Tensor> rotary_cos,
      std::optional<Tensor> rotary_sin, std::optional<Tensor> cache_batch_idx,
      std::optional<Tensor> leftpad_k, std::optional<Tensor> block_table,
      std::optional<Tensor> alibi_slopes, Tensor out, float softmax_scale,
      bool is_causal, int64_t window_size_left, int64_t window_size_right,
      float softcap, bool is_rotary_interleaved,
      int64_t num_splits = 0) const = 0;

 protected:
  Tensor::Size batch_size_{0};

  Tensor::Size seqlen_q_{0};

  Tensor::Size num_heads_{0};

  Tensor::Size head_size_{0};

  Tensor::Size num_kv_heads_{0};

  Tensor::Size page_block_size_{0};

  float softmax_scale_{0.0f};

  bool is_causal_{false};

  int64_t window_size_left_{-1};

  int64_t window_size_right_{-1};

  float softcap_{0.0f};

  bool is_rotary_interleaved_{false};

  int64_t num_splits_{0};

  const DataType q_dtype_;

  bool has_k_{false};

  bool has_v_{false};

  bool has_seqlens_k_{false};

  bool has_rotary_cos_{false};

  bool has_rotary_sin_{false};

  bool has_cache_batch_idx_{false};

  bool has_leftpad_k_{false};

  bool has_block_table_{false};

  bool has_alibi_slopes_{false};
};

}  // namespace infini::ops

#endif
