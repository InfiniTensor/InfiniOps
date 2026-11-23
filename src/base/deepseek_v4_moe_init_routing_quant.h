#ifndef INFINI_OPS_BASE_DEEPSEEK_V4_MOE_INIT_ROUTING_QUANT_H_
#define INFINI_OPS_BASE_DEEPSEEK_V4_MOE_INIT_ROUTING_QUANT_H_

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class DeepseekV4MoeInitRoutingQuant
    : public Operator<DeepseekV4MoeInitRoutingQuant> {
 public:
  static constexpr bool kAutomaticDefault = true;

  DeepseekV4MoeInitRoutingQuant(Tensor expanded_x, Tensor expanded_row_idx,
                                Tensor expert_tokens_count,
                                Tensor dynamic_quant_scale, const Tensor x,
                                const Tensor expert_idx) {}

  virtual void operator()(Tensor expanded_x, Tensor expanded_row_idx,
                          Tensor expert_tokens_count,
                          Tensor dynamic_quant_scale, const Tensor x,
                          const Tensor expert_idx) const = 0;
};

}  // namespace infini::ops

#endif
