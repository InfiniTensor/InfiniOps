#ifndef INFINI_OPS_BASE_LIGHTNING_ATTENTION_INFINILM_H_
#define INFINI_OPS_BASE_LIGHTNING_ATTENTION_INFINILM_H_

#include <cassert>
#include <cstddef>

#include "data_type.h"
#include "operator.h"
#include "tensor.h"

namespace infini::ops {

/// Lightning attention with an indexed recurrent-state pool.
///
/// The operator evaluates the recurrent form of MiniMax-style linear attention
/// with a per-head ALiBi-style decay, one token at a time:
///
///   ratio[h] = exp(-slope[h])
///   S        = ratio[h] * S + outer(k_t[h], v_t[h])
///   out_t[h] = q_t[h] @ S
///
/// The recurrent state of request `b` is read from
/// `initial_state[initial_state_indices[b]]` and the final state is written to
/// `initial_state[final_state_indices[b]]`. The row referenced by
/// `initial_state_indices` is left untouched, so a caller may keep using it.
///
/// Requests are independent and may execute concurrently, so a row used as the
/// destination of one request must not be the source row of another request in
/// the same call.
///
/// This operator is InfiniLM-specific and is therefore classified as custom
/// rather than aligned to an open-source operator. The closest public
/// reference implementation is Flash-Linear-Attention's
/// `fused_recurrent_lightning_attn`; `dexp` there is the per-head decay factor
/// `exp(-slope)` used here.
class LightningAttentionInfinilm
    : public Operator<LightningAttentionInfinilm> {
 public:
  LightningAttentionInfinilm(const Tensor q, const Tensor k, const Tensor v,
                             const Tensor slope, Tensor initial_state,
                             const Tensor initial_state_indices,
                             const Tensor final_state_indices, Tensor out)
      : data_type_{q.dtype()},
        index_dtype_{initial_state_indices.dtype()},
        batch_size_{q.size(0)},
        seq_len_{q.size(1)},
        num_heads_{q.size(2)},
        head_dim_{q.size(3)},
        state_pool_size_{initial_state.size(0)},
        state_pool_stride_{initial_state.stride(0)},
        state_head_stride_{initial_state.stride(1)},
        state_row_stride_{initial_state.stride(2)},
        state_column_stride_{initial_state.stride(3)},
        q_batch_stride_{q.stride(0)},
        q_seq_stride_{q.stride(1)},
        q_head_stride_{q.stride(2)},
        k_batch_stride_{k.stride(0)},
        k_seq_stride_{k.stride(1)},
        k_head_stride_{k.stride(2)},
        v_batch_stride_{v.stride(0)},
        v_seq_stride_{v.stride(1)},
        v_head_stride_{v.stride(2)},
        out_batch_stride_{out.stride(0)},
        out_seq_stride_{out.stride(1)},
        out_head_stride_{out.stride(2)},
        slope_stride_{slope.stride(0)},
        initial_index_stride_{initial_state_indices.stride(0)},
        final_index_stride_{final_state_indices.stride(0)} {
    assert(q.ndim() == 4 && k.ndim() == 4 && v.ndim() == 4 && out.ndim() == 4 &&
           "`LightningAttentionInfinilm` expects [batch, seq, heads, head_dim] tensors");
    assert(q.dtype() == k.dtype() && k.dtype() == v.dtype() &&
           v.dtype() == out.dtype() && out.dtype() == data_type_ &&
           initial_state.dtype() == data_type_ &&
           "`LightningAttentionInfinilm` requires all data tensors to share one dtype");
    assert((data_type_ == DataType::kFloat32 ||
            data_type_ == DataType::kFloat16 ||
            data_type_ == DataType::kBFloat16) &&
           "`LightningAttentionInfinilm` supports float32, float16 and bfloat16");
    assert(q.shape() == k.shape() && k.shape() == v.shape() &&
           v.shape() == out.shape() &&
           "`LightningAttentionInfinilm` requires q, k, v and out to share a shape");
    assert(slope.dtype() == DataType::kFloat32 && slope.ndim() == 1 &&
           slope.size(0) == num_heads_ && slope_stride_ == 1 &&
           "`LightningAttentionInfinilm` expects `slope` to be a contiguous float32 tensor of size num_heads");
    assert(initial_state.ndim() == 4 && initial_state.size(1) == num_heads_ &&
           initial_state.size(2) == head_dim_ &&
           initial_state.size(3) == head_dim_ &&
           "`LightningAttentionInfinilm` expects `initial_state` to be [pool, heads, head_dim, head_dim]");
    assert(state_pool_size_ > 0 && state_column_stride_ == 1 &&
           "`LightningAttentionInfinilm` expects a contiguous state pool on the last dimension");
    assert(initial_state_indices.ndim() == 1 &&
           final_state_indices.ndim() == 1 &&
           initial_state_indices.size(0) == batch_size_ &&
           final_state_indices.size(0) == batch_size_ &&
           "`LightningAttentionInfinilm` expects one state index per request");
    assert(IsIndexDtype(index_dtype_) &&
           final_state_indices.dtype() == index_dtype_ &&
           initial_index_stride_ == 1 && final_index_stride_ == 1 &&
           "`LightningAttentionInfinilm` expects contiguous int32/int64 state indices");
    assert(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1 &&
           out.stride(3) == 1 &&
           "`LightningAttentionInfinilm` requires a contiguous last dimension");
  }

  virtual void operator()(const Tensor q, const Tensor k, const Tensor v,
                          const Tensor slope, Tensor initial_state,
                          const Tensor initial_state_indices,
                          const Tensor final_state_indices,
                          Tensor out) const = 0;

 protected:
  static bool IsIndexDtype(DataType dtype) {
    return dtype == DataType::kInt32 || dtype == DataType::kInt64;
  }

  DataType data_type_;

  DataType index_dtype_;

  Tensor::Size batch_size_{0};

  Tensor::Size seq_len_{0};

  Tensor::Size num_heads_{0};

  Tensor::Size head_dim_{0};

  Tensor::Size state_pool_size_{0};

  Tensor::Stride state_pool_stride_{0};

  Tensor::Stride state_head_stride_{0};

  Tensor::Stride state_row_stride_{0};

  Tensor::Stride state_column_stride_{0};

  Tensor::Stride q_batch_stride_{0};

  Tensor::Stride q_seq_stride_{0};

  Tensor::Stride q_head_stride_{0};

  Tensor::Stride k_batch_stride_{0};

  Tensor::Stride k_seq_stride_{0};

  Tensor::Stride k_head_stride_{0};

  Tensor::Stride v_batch_stride_{0};

  Tensor::Stride v_seq_stride_{0};

  Tensor::Stride v_head_stride_{0};

  Tensor::Stride out_batch_stride_{0};

  Tensor::Stride out_seq_stride_{0};

  Tensor::Stride out_head_stride_{0};

  Tensor::Stride slope_stride_{0};

  Tensor::Stride initial_index_stride_{0};

  Tensor::Stride final_index_stride_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_LIGHTNING_ATTENTION_INFINILM_H_
