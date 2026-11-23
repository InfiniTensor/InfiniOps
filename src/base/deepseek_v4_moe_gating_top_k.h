#ifndef INFINI_OPS_BASE_DEEPSEEK_V4_MOE_GATING_TOP_K_H_
#define INFINI_OPS_BASE_DEEPSEEK_V4_MOE_GATING_TOP_K_H_

#include <optional>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class DeepseekV4MoeGatingTopK : public Operator<DeepseekV4MoeGatingTopK> {
 public:
  static constexpr bool kAutomaticDefault = true;

  DeepseekV4MoeGatingTopK(Tensor topk_weights, Tensor topk_ids,
                          Tensor normalized_scores, const Tensor logits,
                          const std::optional<Tensor> bias,
                          const std::optional<Tensor> input_ids,
                          const std::optional<Tensor> tid2eid, int64_t topk,
                          double routed_scaling_factor) {}

  virtual void operator()(Tensor topk_weights, Tensor topk_ids,
                          Tensor normalized_scores, const Tensor logits,
                          const std::optional<Tensor> bias,
                          const std::optional<Tensor> input_ids,
                          const std::optional<Tensor> tid2eid, int64_t topk,
                          double routed_scaling_factor) const = 0;
};

}  // namespace infini::ops

#endif
