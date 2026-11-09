#ifndef INFINI_OPS_BASE_MOE_ALIGN_H_
#define INFINI_OPS_BASE_MOE_ALIGN_H_

#include <cstdint>

#include "operator.h"

namespace infini::ops {

class MoeAlign : public Operator<MoeAlign> {
 public:
  MoeAlign(Tensor sorted_token_ids, Tensor expert_ids,
           Tensor num_tokens_post_padded, Tensor topk_ids, Tensor expert_map,
           int64_t num_experts, int64_t block_size, bool pad_sorted_token_ids)
      : sorted_token_ids_shape_{sorted_token_ids.shape()},
        expert_ids_shape_{expert_ids.shape()},
        num_tokens_post_padded_shape_{num_tokens_post_padded.shape()},
        topk_ids_shape_{topk_ids.shape()},
        expert_map_shape_{expert_map.shape()},
        num_experts_{num_experts},
        block_size_{block_size},
        pad_sorted_token_ids_{pad_sorted_token_ids},
        numel_{topk_ids.numel()},
        max_num_tokens_padded_{sorted_token_ids.numel()} {
    assert(topk_ids.ndim() == 2 && "`MoeAlign` topk_ids must be a 2D tensor");
    assert(sorted_token_ids.ndim() == 1 &&
           "`MoeAlign` sorted_token_ids must be a 1D tensor");
    assert(expert_ids.ndim() == 1 &&
           "`MoeAlign` expert_ids must be a 1D tensor");
    assert(num_tokens_post_padded.ndim() == 1 &&
           num_tokens_post_padded.numel() == 1 &&
           "`MoeAlign` num_tokens_post_padded must be a scalar tensor");
    assert(topk_ids.dtype() == DataType::kInt32 &&
           "`MoeAlign` topk_ids must be int32");
    assert(sorted_token_ids.dtype() == DataType::kInt32 &&
           "`MoeAlign` sorted_token_ids must be int32");
    assert(expert_ids.dtype() == DataType::kInt32 &&
           "`MoeAlign` expert_ids must be int32");
    assert(num_tokens_post_padded.dtype() == DataType::kInt32 &&
           "`MoeAlign` num_tokens_post_padded must be int32");
    assert(num_experts_ > 0 && "`MoeAlign` num_experts must be positive");
    assert(block_size_ > 0 && "`MoeAlign` block_size must be positive");
    assert(expert_map.numel() == 0 ||
           (expert_map.ndim() == 1 &&
            static_cast<int64_t>(expert_map.numel()) == num_experts_ &&
            expert_map.dtype() == DataType::kInt32) &&
               "`MoeAlign` expert_map must be empty or (num_experts,) int32");
  }

  virtual void operator()(Tensor sorted_token_ids, Tensor expert_ids,
                          Tensor num_tokens_post_padded, Tensor topk_ids,
                          Tensor expert_map, int64_t num_experts,
                          int64_t block_size,
                          bool pad_sorted_token_ids) const = 0;

 protected:
  Tensor::Shape sorted_token_ids_shape_;
  Tensor::Shape expert_ids_shape_;
  Tensor::Shape num_tokens_post_padded_shape_;
  Tensor::Shape topk_ids_shape_;
  Tensor::Shape expert_map_shape_;
  int64_t num_experts_{0};
  int64_t block_size_{0};
  bool pad_sorted_token_ids_{false};
  Tensor::Size numel_{0};
  Tensor::Size max_num_tokens_padded_{0};
};

}  // namespace infini::ops

#endif
