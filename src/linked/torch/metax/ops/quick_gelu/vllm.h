#ifndef INFINI_OPS_LINKED_TORCH_METAX_OPS_QUICK_GELU_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_METAX_OPS_QUICK_GELU_VLLM_H_

#include "linked/torch/ops/quick_gelu.h"
#include "torch/metax/c10.h"

namespace infini::ops::linked::torch::metax {

struct VllmQuickGelu : C10<Device::Type::kMetax> {
  static void Call(at::Tensor& out, at::Tensor& input);
};

}  // namespace infini::ops::linked::torch::metax

namespace infini::ops::linked::torch {

extern template class TorchQuickGelu<
    ::infini::ops::linked::torch::metax::VllmQuickGelu>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<QuickGelu, Device::Type::kMetax, 16>
    : public linked::torch::TorchQuickGelu<
          linked::torch::metax::VllmQuickGelu> {
 public:
  using linked::torch::TorchQuickGelu<
      linked::torch::metax::VllmQuickGelu>::TorchQuickGelu;
  using linked::torch::TorchQuickGelu<
      linked::torch::metax::VllmQuickGelu>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_METAX_OPS_QUICK_GELU_VLLM_H_
