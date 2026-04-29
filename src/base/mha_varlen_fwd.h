#ifndef INFINI_OPS_BASE_MHA_VARLEN_FWD_H_
#define INFINI_OPS_BASE_MHA_VARLEN_FWD_H_

#include <cassert>
#include <optional>

#include "data_type.h"
#include "operator.h"

namespace infini::ops {

// FlashAttention-compatible variable-length forward attention.
//
// Layout follows `flash_attn` `varlen_fwd`:
//   `q`: `[total_q, num_heads, head_size]`.
//   `k` / `v`: `[total_k, num_kv_heads, head_size]`.
//   Paged `k` / `v`: `[num_blocks, block_size, num_kv_heads, head_size]`
//     when `block_table` is supplied.
//   `cu_seqlens_q` / `cu_seqlens_k`: `[batch_size + 1]`.
//
// InfiniOps is an in-place operator API, so `out` must be supplied even
// though FlashAttention can allocate it when omitted.
class MhaVarlenFwd : public Operator<MhaVarlenFwd> {
 public:
  MhaVarlenFwd(const Tensor q, const Tensor k, const Tensor v,
               std::optional<Tensor> out, const Tensor cu_seqlens_q,
               const Tensor cu_seqlens_k, std::optional<Tensor> seqused_k,
               std::optional<Tensor> leftpad_k,
               std::optional<Tensor> block_table,
               std::optional<Tensor> alibi_slopes, int64_t max_seqlen_q,
               int64_t max_seqlen_k, float p_dropout, float softmax_scale,
               bool zero_tensors, bool is_causal, int64_t window_size_left,
               int64_t window_size_right, float softcap, bool return_softmax,
               std::optional<int64_t> generator = std::nullopt,
               int64_t num_splits = 0)
      : batch_size_{cu_seqlens_q.numel() - 1},
        total_q_{q.size(0)},
        num_heads_{q.size(1)},
        head_size_{q.size(2)},
        num_kv_heads_{block_table.has_value() ? k.size(2) : k.size(1)},
        max_seqlen_q_{max_seqlen_q},
        max_seqlen_k_{max_seqlen_k},
        p_dropout_{p_dropout},
        softmax_scale_{softmax_scale},
        zero_tensors_{zero_tensors},
        is_causal_{is_causal},
        window_size_left_{window_size_left},
        window_size_right_{window_size_right},
        softcap_{softcap},
        return_softmax_{return_softmax},
        num_splits_{num_splits},
        q_dtype_{q.dtype()},
        has_out_{out.has_value()},
        has_seqused_k_{seqused_k.has_value()},
        has_leftpad_k_{leftpad_k.has_value()},
        has_block_table_{block_table.has_value()},
        has_alibi_slopes_{alibi_slopes.has_value()},
        has_generator_{generator.has_value()} {
    assert(q.ndim() == 3 &&
           "`MhaVarlenFwd` requires `q` to be `[total_q, heads, dim]`");
    if (has_block_table_) {
      assert(k.ndim() == 4 && v.ndim() == 4 &&
             "`MhaVarlenFwd` with `block_table` requires paged `k` / `v` to "
             "be `[num_blocks, block_size, heads, dim]`.");
      assert(block_table->ndim() == 2 &&
             "`MhaVarlenFwd` requires `block_table` to be 2D.");
      assert(block_table->dtype() == DataType::kInt32 &&
             "`MhaVarlenFwd` requires `block_table` to be `int32`.");
    } else {
      assert(k.ndim() == 3 && v.ndim() == 3 &&
             "`MhaVarlenFwd` requires `k` / `v` to be "
             "`[total_k, heads, dim]`.");
    }
    assert(k.dtype() == q.dtype() && v.dtype() == q.dtype() &&
           "`MhaVarlenFwd` requires `q`, `k`, and `v` to have same dtype");
    assert(k.stride(-1) == 1 && v.stride(-1) == 1 && q.stride(-1) == 1 &&
           "`MhaVarlenFwd` requires contiguous last dimension");
    assert(num_heads_ % num_kv_heads_ == 0 &&
           "`MhaVarlenFwd` requires `num_heads` divisible by `num_kv_heads`");
    assert(head_size_ <= 256 &&
           "`MhaVarlenFwd` supports `head_size` up to 256");
    assert(head_size_ % 8 == 0 &&
           "`MhaVarlenFwd` requires `head_size` to be a multiple of 8");
    assert(cu_seqlens_q.ndim() == 1 && cu_seqlens_k.ndim() == 1 &&
           "`MhaVarlenFwd` requires 1D `cu_seqlens_q` / `cu_seqlens_k`");
    assert(cu_seqlens_q.dtype() == DataType::kInt32 &&
           cu_seqlens_k.dtype() == DataType::kInt32 &&
           "`MhaVarlenFwd` requires `cu_seqlens_q` and `cu_seqlens_k` to be "
           "`int32`.");
    assert(cu_seqlens_q.numel() == cu_seqlens_k.numel() &&
           "`MhaVarlenFwd` requires matching `cu_seqlens` lengths");
    assert(has_out_ && "`MhaVarlenFwd` requires caller-provided `out`");

    if (out.has_value()) {
      assert(out->shape() == q.shape() &&
             "`MhaVarlenFwd` requires `out` to match `q` shape");
      assert(out->stride(-1) == 1 &&
             "`MhaVarlenFwd` requires `out` contiguous last dimension");
    }
  }

  virtual void operator()(
      const Tensor q, const Tensor k, const Tensor v, std::optional<Tensor> out,
      const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
      std::optional<Tensor> seqused_k, std::optional<Tensor> leftpad_k,
      std::optional<Tensor> block_table, std::optional<Tensor> alibi_slopes,
      int64_t max_seqlen_q, int64_t max_seqlen_k, float p_dropout,
      float softmax_scale, bool zero_tensors, bool is_causal,
      int64_t window_size_left, int64_t window_size_right, float softcap,
      bool return_softmax, std::optional<int64_t> generator = std::nullopt,
      int64_t num_splits = 0) const = 0;

 protected:
  Tensor::Size batch_size_{0};

  Tensor::Size total_q_{0};

  Tensor::Size num_heads_{0};

  Tensor::Size head_size_{0};

  Tensor::Size num_kv_heads_{0};

  int64_t max_seqlen_q_{0};

  int64_t max_seqlen_k_{0};

  float p_dropout_{0.0f};

  float softmax_scale_{0.0f};

  bool zero_tensors_{false};

  bool is_causal_{false};

  int64_t window_size_left_{-1};

  int64_t window_size_right_{-1};

  float softcap_{0.0f};

  bool return_softmax_{false};

  int64_t num_splits_{0};

  const DataType q_dtype_;

  bool has_out_{false};

  bool has_seqused_k_{false};

  bool has_leftpad_k_{false};

  bool has_block_table_{false};

  bool has_alibi_slopes_{false};

  bool has_generator_{false};
};

}  // namespace infini::ops

#endif
