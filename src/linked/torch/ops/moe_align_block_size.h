#ifndef INFINI_OPS_LINKED_TORCH_OPS_MOE_ALIGN_BLOCK_SIZE_H_
#define INFINI_OPS_LINKED_TORCH_OPS_MOE_ALIGN_BLOCK_SIZE_H_

#include <optional>

#include "base/moe_align_block_size.h"
#include "torch/tensor_.h"

namespace infini::ops::linked::torch {

template <typename Backend>
class TorchMoeAlignBlockSize : public ::infini::ops::MoeAlignBlockSize {
 public:
  TorchMoeAlignBlockSize(const Tensor topk_ids,
                         const std::optional<Tensor> expert_map,
                         const int64_t num_experts, const int64_t block_size,
                         Tensor sorted_token_ids, Tensor experts_ids,
                         Tensor num_tokens_post_pad)
      : ::infini::ops::MoeAlignBlockSize{topk_ids,           expert_map,
                                         num_experts,        block_size,
                                         sorted_token_ids,   experts_ids,
                                         num_tokens_post_pad},
        device_index_{topk_ids.device().index()} {}

  using ::infini::ops::MoeAlignBlockSize::operator();

  void Run(const Tensor topk_ids, const std::optional<Tensor> expert_map,
           const int64_t num_experts, const int64_t block_size,
           Tensor sorted_token_ids, Tensor experts_ids,
           Tensor num_tokens_post_pad) const override {
    const typename Backend::StreamGuard stream_guard{
        Backend::GetStreamFromExternal(stream_, device_index_)};

    auto at_topk_ids = ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(topk_ids.data()), topk_ids_metadata_.shape(),
        topk_ids_metadata_.strides(), topk_ids_metadata_.dtype(),
        device_index_);
    auto at_sorted_token_ids = ToAtenTensor<Backend::kDeviceType>(
        sorted_token_ids.data(), sorted_token_ids_metadata_.shape(),
        sorted_token_ids_metadata_.strides(),
        sorted_token_ids_metadata_.dtype(), device_index_);
    auto at_experts_ids = ToAtenTensor<Backend::kDeviceType>(
        experts_ids.data(), experts_ids_metadata_.shape(),
        experts_ids_metadata_.strides(), experts_ids_metadata_.dtype(),
        device_index_);
    auto at_num_tokens_post_pad = ToAtenTensor<Backend::kDeviceType>(
        num_tokens_post_pad.data(), num_tokens_post_pad_metadata_.shape(),
        num_tokens_post_pad_metadata_.strides(),
        num_tokens_post_pad_metadata_.dtype(), device_index_);
    std::optional<at::Tensor> at_expert_map;
    if (expert_map.has_value()) {
      at_expert_map.emplace(ToAtenTensor<Backend::kDeviceType>(
          const_cast<void*>(expert_map->data()), expert_map_metadata_->shape(),
          expert_map_metadata_->strides(), expert_map_metadata_->dtype(),
          device_index_));
    }

    Backend::Call(std::move(at_topk_ids), static_cast<int64_t>(num_experts_),
                  static_cast<int64_t>(block_size_),
                  std::move(at_sorted_token_ids), std::move(at_experts_ids),
                  std::move(at_num_tokens_post_pad), std::move(at_expert_map));
  }

 private:
  int device_index_{0};
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_OPS_MOE_ALIGN_BLOCK_SIZE_H_
