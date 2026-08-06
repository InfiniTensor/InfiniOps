#ifndef INFINI_OPS_LINKED_TORCH_METAX_OPS_SILU_AND_MUL_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_METAX_OPS_SILU_AND_MUL_VLLM_H_

#include "linked/torch/ops/silu_and_mul.h"

namespace at {
class Tensor;
}

namespace infini::ops::linked::torch::metax {

struct VllmSiluAndMul {
  static void Call(at::Tensor& out, at::Tensor& input);
};

}  // namespace infini::ops::linked::torch::metax

namespace infini::ops {

template <>
class Operator<SiluAndMul, Device::Type::kMetax, 16>
    : public linked::torch::TorchSiluAndMul<
          Device::Type::kMetax, linked::torch::metax::VllmSiluAndMul> {
 public:
  using linked::torch::TorchSiluAndMul<
      Device::Type::kMetax,
      linked::torch::metax::VllmSiluAndMul>::TorchSiluAndMul;
  using linked::torch::TorchSiluAndMul<
      Device::Type::kMetax, linked::torch::metax::VllmSiluAndMul>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_METAX_OPS_SILU_AND_MUL_VLLM_H_
