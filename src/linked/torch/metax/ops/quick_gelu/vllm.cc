#include "linked/torch/metax/ops/quick_gelu/vllm.h"

// MetaX vLLM `_C` exports this global C++ symbol.
void gelu_quick(at::Tensor& out, at::Tensor& input);

namespace infini::ops::linked::torch::metax {

void VllmQuickGelu::Call(at::Tensor& out, at::Tensor& input) {
  ::gelu_quick(out, input);
}

}  // namespace infini::ops::linked::torch::metax

namespace infini::ops::linked::torch {

template class TorchQuickGelu<
    ::infini::ops::linked::torch::metax::VllmQuickGelu>;

}  // namespace infini::ops::linked::torch
