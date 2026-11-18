#ifndef INFINI_OPS_KUNLUN_TOPK_SOFTMAX_KERNEL_H_
#define INFINI_OPS_KUNLUN_TOPK_SOFTMAX_KERNEL_H_

#include <stdexcept>

#include "base/topk_softmax.h"

namespace infini::ops {

// Placeholder:  has no Kunlun implementation yet. The class must
// exist so downstream consumers (InfiniLM) referencing the operator type
// at compile time build; running it throws.
template <>
class Operator<TopkSoftmax, Device::Type::kKunlun> : public TopkSoftmax {
 public:
  using TopkSoftmax::TopkSoftmax;

  [[noreturn]] void operator()(const Tensor gating_output,
                               std::optional<Tensor> bias,
                               std::optional<Tensor> is_padding,
                               const bool renormalize, Tensor topk_weights,
                               Tensor topk_indices,
                               Tensor token_expert_indices) const override {
    throw std::runtime_error("TopkSoftmax is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
