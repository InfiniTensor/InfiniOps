#ifndef INFINI_OPS_LINKED_TORCH_OPS_FUSED_MARLIN_MOE_H_
#define INFINI_OPS_LINKED_TORCH_OPS_FUSED_MARLIN_MOE_H_

#include <ATen/Functions.h>
#include <ATen/TensorIndexing.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

#include "base/fused_marlin_moe.h"
#include "torch/tensor_.h"

namespace infini::ops::linked::torch {

template <typename Backend>
class TorchFusedMarlinMoe : public ::infini::ops::FusedMarlinMoe {
 public:
  using ::infini::ops::FusedMarlinMoe::FusedMarlinMoe;

  using ::infini::ops::FusedMarlinMoe::operator();

  void operator()(
      const Tensor hidden_states, const Tensor w1, const Tensor w2,
      const Tensor w1_scale, const Tensor w2_scale, const Tensor gating_output,
      const Tensor topk_weights, const Tensor topk_ids,
      const int64_t quant_type_id, const bool apply_router_weight_on_input,
      const int64_t global_num_experts, std::optional<Tensor> expert_map,
      std::optional<Tensor> global_scale1, std::optional<Tensor> global_scale2,
      std::optional<Tensor> g_idx1, std::optional<Tensor> g_idx2,
      std::optional<Tensor> sort_indices1, std::optional<Tensor> sort_indices2,
      std::optional<Tensor> w1_zeros, std::optional<Tensor> w2_zeros,
      std::optional<Tensor> workspace, const bool is_k_full, const bool inplace,
      Tensor out) const override {
    ValidateCallMetadata(hidden_states, w1, w2, w1_scale, w2_scale,
                         gating_output, topk_weights, topk_ids, quant_type_id,
                         apply_router_weight_on_input, global_num_experts,
                         expert_map, global_scale1, global_scale2, g_idx1,
                         g_idx2, sort_indices1, sort_indices2, w1_zeros,
                         w2_zeros, workspace, is_k_full, inplace, out);

    Backend::Validate(quant_type_id,
                      global_scale1.has_value() || global_scale2.has_value(),
                      w1_zeros.has_value(), w2_zeros.has_value());
    const auto aliases = hidden_states.data() == out.data();
    TORCH_CHECK(g_idx1.has_value() == sort_indices1.has_value() &&
                    g_idx2.has_value() == sort_indices2.has_value(),
                "Linked `fused_marlin_moe` requires each `g_idx` with its "
                "`sort_indices`.");
    TORCH_CHECK(aliases == inplace,
                "Linked `fused_marlin_moe` `out` must alias `hidden_states` "
                "exactly when `inplace` is true.");
    TORCH_CHECK(expert_map.has_value() || global_num_experts == -1 ||
                    global_num_experts == static_cast<int64_t>(w1.size(0)),
                "Linked `fused_marlin_moe` requires `global_num_experts` to "
                "match local experts when `expert_map` is absent.");

    const typename Backend::StreamGuard stream_guard{
        Backend::GetStreamFromExternal(stream_, device_index_)};

    auto at_hidden_states = ToAten(hidden_states);
    auto at_w1 = ToAten(w1);
    auto at_w2 = ToAten(w2);
    auto at_w1_scale = ToAten(w1_scale);
    auto at_w2_scale = ToAten(w2_scale);
    [[maybe_unused]] auto at_gating_output = ToAten(gating_output);
    auto at_topk_weights = ToAten(topk_weights);
    auto at_topk_ids = ToAten(topk_ids);
    auto at_expert_map = ToOptionalAten(expert_map);
    auto at_global_scale1 = ToOptionalAten(global_scale1);
    auto at_global_scale2 = ToOptionalAten(global_scale2);
    auto at_g_idx1 = ToOptionalAten(g_idx1);
    auto at_g_idx2 = ToOptionalAten(g_idx2);
    auto at_sort_indices1 = ToOptionalAten(sort_indices1);
    auto at_sort_indices2 = ToOptionalAten(sort_indices2);
    auto at_w1_zeros = ToOptionalAten(w1_zeros);
    auto at_w2_zeros = ToOptionalAten(w2_zeros);
    auto at_workspace_input = ToOptionalAten(workspace);
    auto at_out = ToAten(out);

    const auto num_tokens = at_hidden_states.size(0);
    const auto hidden_size = at_hidden_states.size(1);
    const auto num_experts = at_w1.size(0);
    const auto topk = at_topk_ids.size(1);
    const auto intermediate_size = at_w2.size(1) * 16;
    const auto route_count = num_tokens * topk;
    const auto block_size = SelectBlockSize(num_tokens, topk, num_experts);
    const auto routed_num_experts =
        global_num_experts == -1 ? num_experts : global_num_experts;

    const auto max_num_tokens_padded =
        route_count + routed_num_experts * (block_size - 1);
    auto sorted_token_ids = at::empty({max_num_tokens_padded},
                                      at_topk_ids.options().dtype(at::kInt));
    auto expert_ids =
        at::empty({(max_num_tokens_padded + block_size - 1) / block_size},
                  at_topk_ids.options().dtype(at::kInt));
    auto num_tokens_post_padded =
        at::empty({1}, at_topk_ids.options().dtype(at::kInt));
    Backend::CallAlign(at_topk_ids, routed_num_experts, block_size,
                       sorted_token_ids, expert_ids, num_tokens_post_padded);
    if (at_expert_map) {
      expert_ids = at_expert_map->index({expert_ids.to(at::kLong)});
    }

    at::Tensor at_workspace;
    if (at_workspace_input) {
      at_workspace = *at_workspace_input;
      TORCH_CHECK(
          at_workspace.numel() >= Backend::WorkspaceSize(device_index_),
          "Linked `fused_marlin_moe` requires an int32 workspace with at "
          "least four entries per streaming multiprocessor.");
    } else {
      at_workspace = at::zeros({Backend::WorkspaceSize(device_index_)},
                               at_hidden_states.options().dtype(at::kInt));
    }

    const auto cache13_size =
        route_count * std::max(intermediate_size * 2, hidden_size);
    auto cache13 = at::empty({cache13_size}, at_hidden_states.options());
    auto cache1 = cache13.narrow(0, 0, route_count * intermediate_size * 2)
                      .view({route_count, intermediate_size * 2});
    auto cache2 =
        at::empty({route_count, intermediate_size}, at_hidden_states.options());
    auto cache3 = cache13.narrow(0, 0, route_count * hidden_size)
                      .view({route_count, hidden_size});

    const auto is_ep = at_expert_map.has_value();
    const auto use_atomic_add =
        Backend::UseAtomicAdd(at_hidden_states.scalar_type(), device_index_);
    Backend::CallMarlin(at_hidden_states, cache1, at_w1, at_w1_scale,
                        at_global_scale1, at_w1_zeros, at_g_idx1,
                        at_sort_indices1, at_workspace, sorted_token_ids,
                        expert_ids, num_tokens_post_padded, at_topk_weights,
                        block_size, topk, apply_router_weight_on_input, is_ep,
                        quant_type_id, num_tokens, intermediate_size * 2,
                        hidden_size, is_k_full, use_atomic_add, true, false);

    const auto cache1_left = cache1.narrow(1, 0, intermediate_size);
    const auto cache1_right =
        cache1.narrow(1, intermediate_size, intermediate_size);
    at::silu_out(cache2, cache1_left);
    cache2.mul_(cache1_right);

    if (is_ep) {
      cache3.zero_();
    }
    Backend::CallMarlin(
        cache2, cache3, at_w2, at_w2_scale, at_global_scale2, at_w2_zeros,
        at_g_idx2, at_sort_indices2, at_workspace, sorted_token_ids, expert_ids,
        num_tokens_post_padded, at_topk_weights, block_size, 1,
        !apply_router_weight_on_input, is_ep, quant_type_id, route_count,
        hidden_size, intermediate_size, is_k_full, use_atomic_add, true, false);

    const auto routed_output = cache3.view({num_tokens, topk, hidden_size});
    at::sum_out(at_out, routed_output, {1}, false, std::nullopt);
  }

 private:
  static int64_t SelectBlockSize(const int64_t num_tokens, const int64_t topk,
                                 const int64_t num_experts) {
    int64_t block_size = 64;
    for (const auto candidate : std::array<int64_t, 5>{8, 16, 32, 48, 64}) {
      block_size = candidate;
      if (static_cast<double>(num_tokens * topk) /
              static_cast<double>(num_experts) /
              static_cast<double>(block_size) <
          0.9) {
        break;
      }
    }

    return block_size;
  }

  at::Tensor ToAten(const Tensor tensor) const {
    return ToAtenTensor<Backend::kDeviceType>(const_cast<void*>(tensor.data()),
                                              tensor.shape(), tensor.strides(),
                                              tensor.dtype(), device_index_);
  }

  std::optional<at::Tensor> ToOptionalAten(
      const std::optional<Tensor>& tensor) const {
    if (!tensor) {
      return std::nullopt;
    }

    return ToAten(*tensor);
  }
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_OPS_FUSED_MARLIN_MOE_H_
