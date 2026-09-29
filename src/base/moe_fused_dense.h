#ifndef INFINI_OPS_BASE_MOE_FUSED_DENSE_H_
#define INFINI_OPS_BASE_MOE_FUSED_DENSE_H_

#include <cstdint>

#include "operator.h"

namespace infini::ops {

class MoeFusedDense : public Operator<MoeFusedDense> {
 public:
  MoeFusedDense(Tensor output, Tensor hidden_states, Tensor w13, Tensor w2,
                Tensor topk_weights, Tensor topk_ids, Tensor sorted_token_ids,
                Tensor expert_ids, Tensor num_tokens_post_padded)
      : output_shape_{output.shape()},
        hidden_states_shape_{hidden_states.shape()},
        w13_shape_{w13.shape()},
        w2_shape_{w2.shape()},
        topk_weights_shape_{topk_weights.shape()},
        topk_ids_shape_{topk_ids.shape()},
        sorted_token_ids_shape_{sorted_token_ids.shape()},
        expert_ids_shape_{expert_ids.shape()},
        num_tokens_post_padded_shape_{num_tokens_post_padded.shape()},
        num_tokens_{hidden_states.size(0)},
        hidden_size_{hidden_states.size(1)},
        num_experts_{w13.size(0)},
        intermediate_size_{w2.size(2)},
        topk_{static_cast<int64_t>(topk_ids.size(1))},
        max_num_tokens_padded_{sorted_token_ids.numel()},
        max_num_blocks_{expert_ids.numel()},
        dtype_{output.dtype()} {
    assert(output.ndim() == 2 && hidden_states.ndim() == 2 &&
           "`MoeFusedDense` output and hidden_states must be 2D tensors");
    assert(w13.ndim() == 3 && w2.ndim() == 3 &&
           "`MoeFusedDense` w13 and w2 must be 3D tensors");
    assert(topk_weights.ndim() == 2 && topk_ids.ndim() == 2 &&
           "`MoeFusedDense` topk_weights and topk_ids must be 2D tensors");
    assert(
        sorted_token_ids.ndim() == 1 && expert_ids.ndim() == 1 &&
        "`MoeFusedDense` sorted_token_ids and expert_ids must be 1D tensors");
    assert(num_tokens_post_padded.ndim() == 1 &&
           num_tokens_post_padded.numel() == 1 &&
           "`MoeFusedDense` num_tokens_post_padded must be a scalar tensor");
    assert(
        output.dtype() == hidden_states.dtype() &&
        output.dtype() == w13.dtype() && output.dtype() == w2.dtype() &&
        "`MoeFusedDense` all weight/output tensors must have the same dtype");
    assert(topk_weights.dtype() == DataType::kFloat32 &&
           "`MoeFusedDense` topk_weights must be float32");
    assert(topk_ids.dtype() == DataType::kInt32 &&
           sorted_token_ids.dtype() == DataType::kInt32 &&
           expert_ids.dtype() == DataType::kInt32 &&
           num_tokens_post_padded.dtype() == DataType::kInt32 &&
           "`MoeFusedDense` index tensors must be int32");
    assert(output.size(0) == num_tokens_ && output.size(1) == hidden_size_ &&
           "`MoeFusedDense` output shape must be (num_tokens, hidden_size)");
    assert(w13.size(2) == hidden_size_ &&
           "`MoeFusedDense` w13 must have shape (num_experts, w13_rows, "
           "hidden_size)");
    assert(w2.size(0) == num_experts_ && w2.size(1) == hidden_size_ &&
           "`MoeFusedDense` w2 must have shape (num_experts, hidden_size, "
           "intermediate_size)");
    assert(topk_weights.size(0) == num_tokens_ &&
           topk_weights.size(1) == static_cast<Tensor::Size>(topk_) &&
           topk_ids.size(0) == num_tokens_ &&
           "`MoeFusedDense` topk shapes must match");
    assert(max_num_tokens_padded_ >= num_tokens_ * topk_ &&
           max_num_blocks_ > 0 &&
           "`MoeFusedDense` invalid sorted_token_ids or expert_ids sizes");
  }

  virtual void operator()(Tensor output, Tensor hidden_states, Tensor w13,
                          Tensor w2, Tensor topk_weights, Tensor topk_ids,
                          Tensor sorted_token_ids, Tensor expert_ids,
                          Tensor num_tokens_post_padded) const = 0;

 protected:
  Tensor::Shape output_shape_;
  Tensor::Shape hidden_states_shape_;
  Tensor::Shape w13_shape_;
  Tensor::Shape w2_shape_;
  Tensor::Shape topk_weights_shape_;
  Tensor::Shape topk_ids_shape_;
  Tensor::Shape sorted_token_ids_shape_;
  Tensor::Shape expert_ids_shape_;
  Tensor::Shape num_tokens_post_padded_shape_;
  Tensor::Size num_tokens_{0};
  Tensor::Size hidden_size_{0};
  Tensor::Size num_experts_{0};
  Tensor::Size intermediate_size_{0};
  int64_t topk_{0};
  Tensor::Size max_num_tokens_padded_{0};
  Tensor::Size max_num_blocks_{0};
  DataType dtype_;
};

}  // namespace infini::ops

#endif
