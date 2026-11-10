#ifndef INFINI_OPS_LINKED_TORCH_METAX_OPS_TOPK_SIGMOID_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_METAX_OPS_TOPK_SIGMOID_VLLM_H_

#include <optional>

#include "linked/torch/ops/topk_sigmoid.h"
#include "torch/metax/c10.h"

namespace infini::ops::linked::torch::metax {

struct VllmTopkSigmoid : C10<Device::Type::kMetax> {
  static void Validate(bool has_is_padding, double routed_scaling_factor);

  static void Call(at::Tensor topk_weights, at::Tensor topk_indices,
                   at::Tensor token_expert_indices, at::Tensor gating_output,
                   bool renormalize, std::optional<at::Tensor> bias);
};

}  // namespace infini::ops::linked::torch::metax

namespace infini::ops::linked::torch {

extern template class TorchTopkSigmoid<
    ::infini::ops::linked::torch::metax::VllmTopkSigmoid>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<TopkSigmoid, Device::Type::kMetax, 16>
    : public linked::torch::TorchTopkSigmoid<
          linked::torch::metax::VllmTopkSigmoid> {
 public:
  using linked::torch::TorchTopkSigmoid<
      linked::torch::metax::VllmTopkSigmoid>::TorchTopkSigmoid;
  using linked::torch::TorchTopkSigmoid<
      linked::torch::metax::VllmTopkSigmoid>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_METAX_OPS_TOPK_SIGMOID_VLLM_H_
