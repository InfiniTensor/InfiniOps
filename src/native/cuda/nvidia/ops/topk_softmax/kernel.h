#ifndef INFINI_OPS_NVIDIA_TOPK_SOFTMAX_KERNEL_H_
#define INFINI_OPS_NVIDIA_TOPK_SOFTMAX_KERNEL_H_

#include <optional>

#include "base/topk_softmax.h"

namespace infini::ops {

template <>
class Operator<TopkSoftmax, Device::Type::kNvidia, 0> : public TopkSoftmax {
 public:
  using TopkSoftmax::TopkSoftmax;

  void operator()(const Tensor gating_output, std::optional<Tensor> bias,
                  std::optional<Tensor> is_padding, const bool renormalize,
                  Tensor topk_weights, Tensor topk_indices,
                  Tensor token_expert_indices) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_TOPK_SOFTMAX_KERNEL_H_
