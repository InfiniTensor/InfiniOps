#ifndef INFINI_OPS_KUNLUN_TOPK_SIGMOID_KERNEL_H_
#define INFINI_OPS_KUNLUN_TOPK_SIGMOID_KERNEL_H_

#include <stdexcept>

#include "base/topk_sigmoid.h"

namespace infini::ops {

// Placeholder:  has no Kunlun implementation yet. The class must
// exist so downstream consumers (InfiniLM) referencing the operator type
// at compile time build; running it throws.
template <>
class Operator<TopkSigmoid, Device::Type::kKunlun> : public TopkSigmoid {
 public:
  using TopkSigmoid::TopkSigmoid;

  [[noreturn]] void operator()(const Tensor gating_output,
                               std::optional<Tensor> e_score_correction_bias,
                               std::optional<Tensor> is_padding,
                               const bool renormalize,
                               const double routed_scaling_factor,
                               Tensor topk_weights, Tensor topk_ids,
                               Tensor token_expert_indices) const override {
    throw std::runtime_error("TopkSigmoid is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
