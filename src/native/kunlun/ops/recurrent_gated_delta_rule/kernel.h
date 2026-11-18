#ifndef INFINI_OPS_KUNLUN_RECURRENT_GATED_DELTA_RULE_KERNEL_H_
#define INFINI_OPS_KUNLUN_RECURRENT_GATED_DELTA_RULE_KERNEL_H_

#include <stdexcept>

#include "base/recurrent_gated_delta_rule.h"

namespace infini::ops {

// Placeholder:  has no Kunlun implementation yet. The class must
// exist so downstream consumers (InfiniLM) referencing the operator type
// at compile time build; running it throws.
template <>
class Operator<RecurrentGatedDeltaRule, Device::Type::kKunlun>
    : public RecurrentGatedDeltaRule {
 public:
  using RecurrentGatedDeltaRule::RecurrentGatedDeltaRule;

  [[noreturn]] void operator()(
      const Tensor q, const Tensor k, const Tensor v, const Tensor g,
      const Tensor beta, Tensor initial_state,
      const std::optional<Tensor> initial_state_indices,
      const std::optional<Tensor> final_state_indices, Tensor out,
      std::optional<Tensor> final_state, bool use_qk_l2norm) const override {
    throw std::runtime_error(
        "RecurrentGatedDeltaRule is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
