#ifndef INFINI_OPS_BASE_MOE_TOPK_SOFTMAX_H_
#define INFINI_OPS_BASE_MOE_TOPK_SOFTMAX_H_

#include <cstdint>

#include "operator.h"

namespace infini::ops {

class MoeTopkSoftmax : public Operator<MoeTopkSoftmax> {
 public:
  MoeTopkSoftmax(Tensor topk_weights, Tensor topk_indices, Tensor gating_output,
                 Tensor correction_bias, bool renormalize,
                 float moe_softcapping)
      : topk_weights_shape_{topk_weights.shape()},
        topk_indices_shape_{topk_indices.shape()},
        gating_output_shape_{gating_output.shape()},
        correction_bias_shape_{correction_bias.shape()},
        renormalize_{renormalize},
        moe_softcapping_{moe_softcapping},
        num_tokens_{gating_output.size(0)},
        num_experts_{gating_output.size(1)},
        topk_{static_cast<int64_t>(topk_weights.size(1))},
        has_correction_bias_{correction_bias.numel() > 0} {
    assert(gating_output.ndim() == 2 &&
           "`MoeTopkSoftmax` gating_output must be a 2D tensor");
    assert(topk_weights.ndim() == 2 &&
           "`MoeTopkSoftmax` topk_weights must be a 2D tensor");
    assert(topk_indices.ndim() == 2 &&
           "`MoeTopkSoftmax` topk_indices must be a 2D tensor");
    assert(topk_weights.dtype() == DataType::kFloat32 &&
           "`MoeTopkSoftmax` topk_weights must be float32");
    assert(topk_indices.dtype() == DataType::kInt32 &&
           "`MoeTopkSoftmax` topk_indices must be int32");
    assert(gating_output.dtype() == DataType::kFloat16 ||
           gating_output.dtype() == DataType::kBFloat16 ||
           gating_output.dtype() == DataType::kFloat32 &&
               "`MoeTopkSoftmax` gating_output must be fp16/bf16/fp32");
    assert(topk_ > 0 && topk_ <= static_cast<int64_t>(num_experts_) &&
           "`MoeTopkSoftmax` topk must be in (0, num_experts]");
    assert(topk_weights.size(0) == static_cast<Tensor::Size>(num_tokens_) &&
           topk_indices.size(0) == static_cast<Tensor::Size>(num_tokens_) &&
           topk_indices.size(1) == static_cast<Tensor::Size>(topk_) &&
           "`MoeTopkSoftmax` output shapes must match");
    assert(
        correction_bias.numel() == 0 ||
        (correction_bias.ndim() == 1 &&
         correction_bias.numel() == static_cast<Tensor::Size>(num_experts_) &&
         correction_bias.dtype() == DataType::kFloat32) &&
            "`MoeTopkSoftmax` correction_bias must be empty or (num_experts,) "
            "float32");
  }

  virtual void operator()(Tensor topk_weights, Tensor topk_indices,
                          Tensor gating_output, Tensor correction_bias,
                          bool renormalize, float moe_softcapping) const = 0;

 protected:
  Tensor::Shape topk_weights_shape_;
  Tensor::Shape topk_indices_shape_;
  Tensor::Shape gating_output_shape_;
  Tensor::Shape correction_bias_shape_;
  bool renormalize_{false};
  float moe_softcapping_{0.0f};
  Tensor::Size num_tokens_{0};
  Tensor::Size num_experts_{0};
  int64_t topk_{0};
  bool has_correction_bias_{false};
};

}  // namespace infini::ops

#endif
