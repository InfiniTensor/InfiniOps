#ifndef INFINI_OPS_LINKED_TORCH_METAX_OPS_MOE_ALIGN_BLOCK_SIZE_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_METAX_OPS_MOE_ALIGN_BLOCK_SIZE_VLLM_H_

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>

#include <cassert>
#include <utility>

#include "linked/torch/ops/moe_align_block_size.h"
#include "torch/metax/c10.h"

namespace infini::ops::linked::torch::metax {

struct VllmMoeAlignBlockSize : C10<Device::Type::kMetax> {
  static void Call(at::Tensor topk_ids, int64_t num_experts, int64_t block_size,
                   at::Tensor sorted_token_ids, at::Tensor experts_ids,
                   at::Tensor num_tokens_post_pad,
                   std::optional<at::Tensor> maybe_expert_map) {
    static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
        "_moe_C::moe_align_block_size", "");
    c10::Stack stack;
    stack.reserve(7);
    stack.emplace_back(std::move(topk_ids));
    stack.emplace_back(num_experts);
    stack.emplace_back(block_size);
    stack.emplace_back(std::move(sorted_token_ids));
    stack.emplace_back(std::move(experts_ids));
    stack.emplace_back(std::move(num_tokens_post_pad));
    if (maybe_expert_map.has_value()) {
      stack.emplace_back(std::move(*maybe_expert_map));
    } else {
      stack.emplace_back();
    }
    op.callBoxed(&stack);

    assert(stack.empty() &&
           "`moe_align_block_size` returned an unexpected number of values");
  }
};

}  // namespace infini::ops::linked::torch::metax

namespace infini::ops::linked::torch {

extern template class TorchMoeAlignBlockSize<
    ::infini::ops::linked::torch::metax::VllmMoeAlignBlockSize>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<MoeAlignBlockSize, Device::Type::kMetax, 0>
    : public linked::torch::TorchMoeAlignBlockSize<
          linked::torch::metax::VllmMoeAlignBlockSize> {
 public:
  using linked::torch::TorchMoeAlignBlockSize<
      linked::torch::metax::VllmMoeAlignBlockSize>::TorchMoeAlignBlockSize;
  using linked::torch::TorchMoeAlignBlockSize<
      linked::torch::metax::VllmMoeAlignBlockSize>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_METAX_OPS_MOE_ALIGN_BLOCK_SIZE_VLLM_H_
