#include "linked/torch/metax/ops/silu_and_mul/vllm.h"

#include "linked/torch/metax/torch_context.h"
#include "linked/torch/ops/silu_and_mul_impl.h"

// MetaX vLLM `_C` exports this global C++ symbol.
void silu_and_mul(at::Tensor& out, at::Tensor& input);

namespace infini::ops::linked::torch::metax {

void VllmSiluAndMul::Call(at::Tensor& out, at::Tensor& input) {
  ::silu_and_mul(out, input);
}

}  // namespace infini::ops::linked::torch::metax

namespace infini::ops::linked::torch {

template class TorchSiluAndMul<
    Device::Type::kMetax, ::infini::ops::linked::torch::metax::VllmSiluAndMul>;

}  // namespace infini::ops::linked::torch
