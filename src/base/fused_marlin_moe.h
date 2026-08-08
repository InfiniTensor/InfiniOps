#ifndef INFINI_OPS_BASE_FUSED_MARLIN_MOE_H_
#define INFINI_OPS_BASE_FUSED_MARLIN_MOE_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

#include "operator.h"

namespace infini::ops {

// Aligned with vLLM `fused_marlin_moe` at commit
// bcc0a3cbefe55f99da4821f9d89106e3d71e4867.
class FusedMarlinMoe : public Operator<FusedMarlinMoe> {
 public:
  FusedMarlinMoe(
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
      Tensor out)
      : hidden_states_metadata_{hidden_states},
        w1_metadata_{w1},
        w2_metadata_{w2},
        w1_scale_metadata_{w1_scale},
        w2_scale_metadata_{w2_scale},
        gating_output_metadata_{gating_output},
        topk_weights_metadata_{topk_weights},
        topk_ids_metadata_{topk_ids},
        expert_map_metadata_{expert_map},
        global_scale1_metadata_{global_scale1},
        global_scale2_metadata_{global_scale2},
        g_idx1_metadata_{g_idx1},
        g_idx2_metadata_{g_idx2},
        sort_indices1_metadata_{sort_indices1},
        sort_indices2_metadata_{sort_indices2},
        w1_zeros_metadata_{w1_zeros},
        w2_zeros_metadata_{w2_zeros},
        workspace_metadata_{workspace},
        out_metadata_{out},
        quant_type_id_{quant_type_id},
        apply_router_weight_on_input_{apply_router_weight_on_input},
        global_num_experts_{global_num_experts},
        is_k_full_{is_k_full},
        inplace_{inplace},
        device_index_{hidden_states.device().index()} {
    Validate(hidden_states, w1, w2, w1_scale, w2_scale, gating_output,
             topk_weights, topk_ids, expert_map, global_scale1, global_scale2,
             g_idx1, g_idx2, sort_indices1, sort_indices2, w1_zeros, w2_zeros,
             workspace, out);
  }

  virtual void operator()(
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
      Tensor out) const = 0;

 protected:
  void ValidateCallMetadata(
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
      Tensor out) const {
    assert(quant_type_id == quant_type_id_ &&
           apply_router_weight_on_input == apply_router_weight_on_input_ &&
           global_num_experts == global_num_experts_ &&
           is_k_full == is_k_full_ && inplace == inplace_ &&
           "`FusedMarlinMoe` attributes changed after descriptor creation");

    const std::equal_to<Tensor> same_metadata;
    const auto optional_matches = [&](const std::optional<Tensor>& expected,
                                      const std::optional<Tensor>& actual) {
      return expected.has_value() == actual.has_value() &&
             (!expected || same_metadata(*expected, *actual));
    };
    const auto matches =
        same_metadata(hidden_states_metadata_, hidden_states) &&
        same_metadata(w1_metadata_, w1) && same_metadata(w2_metadata_, w2) &&
        same_metadata(w1_scale_metadata_, w1_scale) &&
        same_metadata(w2_scale_metadata_, w2_scale) &&
        same_metadata(gating_output_metadata_, gating_output) &&
        same_metadata(topk_weights_metadata_, topk_weights) &&
        same_metadata(topk_ids_metadata_, topk_ids) &&
        optional_matches(expert_map_metadata_, expert_map) &&
        optional_matches(global_scale1_metadata_, global_scale1) &&
        optional_matches(global_scale2_metadata_, global_scale2) &&
        optional_matches(g_idx1_metadata_, g_idx1) &&
        optional_matches(g_idx2_metadata_, g_idx2) &&
        optional_matches(sort_indices1_metadata_, sort_indices1) &&
        optional_matches(sort_indices2_metadata_, sort_indices2) &&
        optional_matches(w1_zeros_metadata_, w1_zeros) &&
        optional_matches(w2_zeros_metadata_, w2_zeros) &&
        optional_matches(workspace_metadata_, workspace) &&
        same_metadata(out_metadata_, out);
    assert(matches && "`FusedMarlinMoe` call metadata must match descriptor");

    const auto aliases = hidden_states.data() == out.data();
    assert(aliases == inplace &&
           "`FusedMarlinMoe` `out` alias does not match `inplace`");
  }

 private:
  void Validate(const Tensor hidden_states, const Tensor w1, const Tensor w2,
                const Tensor w1_scale, const Tensor w2_scale,
                const Tensor gating_output, const Tensor topk_weights,
                const Tensor topk_ids, std::optional<Tensor> expert_map,
                std::optional<Tensor> global_scale1,
                std::optional<Tensor> global_scale2,
                std::optional<Tensor> g_idx1, std::optional<Tensor> g_idx2,
                std::optional<Tensor> sort_indices1,
                std::optional<Tensor> sort_indices2,
                std::optional<Tensor> w1_zeros, std::optional<Tensor> w2_zeros,
                std::optional<Tensor> workspace, const Tensor out) const {
    assert(hidden_states.ndim() == 2 && hidden_states.size(0) > 0 &&
           hidden_states.size(1) > 0 && hidden_states.IsContiguous() &&
           (hidden_states.dtype() == DataType::kFloat16 ||
            hidden_states.dtype() == DataType::kBFloat16) &&
           "`FusedMarlinMoe` requires non-empty contiguous float16 or "
           "bfloat16 `hidden_states`");

    constexpr int64_t kUint4 = 1125899906843648;
    constexpr int64_t kUint4B8 = 1125899907892224;
    [[maybe_unused]] constexpr int64_t kUint8B128 = 1125899923621888;
    [[maybe_unused]] constexpr int64_t kFloat8E4M3Fn = 2814749767172868;
    constexpr int64_t kFloat4E2M1F = 562949953487106;
    const auto num_bits = quant_type_id_ == kUint4 ||
                                  quant_type_id_ == kUint4B8 ||
                                  quant_type_id_ == kFloat4E2M1F
                              ? 4
                              : 8;
    assert((quant_type_id_ == kUint4 || quant_type_id_ == kUint4B8 ||
            quant_type_id_ == kUint8B128 || quant_type_id_ == kFloat8E4M3Fn ||
            quant_type_id_ == kFloat4E2M1F) &&
           "`FusedMarlinMoe` received an unsupported `quant_type_id`");

    const auto num_tokens = hidden_states.size(0);
    const auto hidden_size = hidden_states.size(1);
    const auto num_experts = w1.ndim() == 3 ? w1.size(0) : 0;
    const auto topk = topk_ids.ndim() == 2 ? topk_ids.size(1) : 0;
    const auto intermediate_size = w2.ndim() == 3 ? w2.size(1) * 16 : 0;
    [[maybe_unused]] const auto pack_factor = 32 / num_bits;

    assert(num_experts > 0 && num_experts < 1024 && topk > 0 &&
           hidden_size % 64 == 0 && intermediate_size > 0 &&
           intermediate_size % 32 == 0 &&
           intermediate_size <= std::numeric_limits<Tensor::Size>::max() / 32 &&
           "`FusedMarlinMoe` received unsupported dimensions");
    assert(w1.ndim() == 3 && w2.ndim() == 3 && w1.size(1) * 16 == hidden_size &&
           w1.size(2) == intermediate_size * 2 * 16 / pack_factor &&
           w2.size(0) == num_experts &&
           w2.size(2) == hidden_size * 16 / pack_factor &&
           w1.dtype() == DataType::kInt32 && w2.dtype() == DataType::kInt32 &&
           w1.IsContiguous() && w2.IsContiguous() &&
           "`FusedMarlinMoe` received invalid packed weights");

    assert(w1_scale.ndim() == 3 && w2_scale.ndim() == 3 &&
           w1_scale.size(0) == num_experts && w1_scale.size(1) > 0 &&
           w1_scale.size(2) == intermediate_size * 2 &&
           hidden_size % w1_scale.size(1) == 0 &&
           w2_scale.size(0) == num_experts && w2_scale.size(1) > 0 &&
           w2_scale.size(2) == hidden_size &&
           intermediate_size % w2_scale.size(1) == 0 &&
           w1_scale.dtype() == hidden_states.dtype() &&
           w2_scale.dtype() == hidden_states.dtype() &&
           w1_scale.IsContiguous() && w2_scale.IsContiguous() &&
           "`FusedMarlinMoe` received invalid weight scales");

    assert(gating_output.ndim() > 0 && gating_output.size(0) == num_tokens &&
           "`FusedMarlinMoe` `gating_output` token count must match "
           "`hidden_states`");
    assert(topk_ids.shape() == topk_weights.shape() &&
           topk_ids.size(0) == num_tokens &&
           topk_ids.dtype() == DataType::kInt32 &&
           topk_weights.dtype() == DataType::kFloat32 &&
           topk_ids.IsContiguous() && topk_weights.IsContiguous() &&
           "`FusedMarlinMoe` received invalid top-k routing tensors");

    [[maybe_unused]] const auto routed_num_experts =
        global_num_experts_ == -1 ? num_experts : global_num_experts_;
    assert(routed_num_experts > 0 && routed_num_experts < 1024 &&
           "`FusedMarlinMoe` requires `global_num_experts` in `[1, 1023]` "
           "or `-1`");
    assert((expert_map || routed_num_experts == num_experts) &&
           "`FusedMarlinMoe` requires `global_num_experts` to match local "
           "experts when `expert_map` is absent");
    if (expert_map) {
      assert(expert_map->ndim() == 1 &&
             expert_map->numel() == routed_num_experts &&
             expert_map->dtype() == DataType::kInt32 &&
             expert_map->IsContiguous() &&
             "`FusedMarlinMoe` received an invalid `expert_map`");
    }

    [[maybe_unused]] const auto optional_pair =
        [](const std::optional<Tensor>& first,
           const std::optional<Tensor>& second) {
          return first.has_value() == second.has_value();
        };
    assert(optional_pair(g_idx1, sort_indices1) &&
           optional_pair(g_idx2, sort_indices2) &&
           "`FusedMarlinMoe` requires each `g_idx` with its "
           "`sort_indices`");

    [[maybe_unused]] const auto has_zero_points = quant_type_id_ == kUint4;
    const auto validate_layer_metadata =
        [&](const std::optional<Tensor>& zeros,
            const std::optional<Tensor>& g_idx,
            const std::optional<Tensor>& sort_indices, const Tensor scales,
            const Tensor::Size size_k, const Tensor::Size size_n) {
          assert(zeros.has_value() == has_zero_points &&
                 "`FusedMarlinMoe` zero points do not match the "
                 "quantization type");
          if (zeros) {
            assert(zeros->ndim() == 3 && zeros->size(0) == num_experts &&
                   zeros->size(1) == scales.size(1) &&
                   zeros->size(2) == size_n / pack_factor &&
                   zeros->dtype() == DataType::kInt32 &&
                   "`FusedMarlinMoe` received invalid zero points");
          }
          if (g_idx && sort_indices) {
            const auto both_empty =
                g_idx->numel() == 0 && sort_indices->numel() == 0;
            const auto valid_nonempty =
                g_idx->ndim() == 2 && g_idx->size(0) == num_experts &&
                g_idx->size(1) == size_k &&
                sort_indices->shape() == g_idx->shape() &&
                (!is_k_full_ || scales.size(1) > 1);
            assert(g_idx->dtype() == DataType::kInt32 &&
                   sort_indices->dtype() == DataType::kInt32 &&
                   (both_empty || valid_nonempty) &&
                   "`FusedMarlinMoe` received invalid activation-order "
                   "metadata");
          }
        };
    validate_layer_metadata(w1_zeros, g_idx1, sort_indices1, w1_scale,
                            hidden_size, intermediate_size * 2);
    validate_layer_metadata(w2_zeros, g_idx2, sort_indices2, w2_scale,
                            intermediate_size, hidden_size);

    const auto same_device = [&](const Tensor tensor) {
      return tensor.device().type() == hidden_states.device().type() &&
             tensor.device().index() == hidden_states.device().index();
    };
    [[maybe_unused]] const auto valid_optional =
        [&](const std::optional<Tensor>& tensor) {
          return !tensor || (same_device(*tensor) && tensor->IsContiguous());
        };
    assert(same_device(w1) && same_device(w2) && same_device(w1_scale) &&
           same_device(w2_scale) && same_device(gating_output) &&
           same_device(topk_weights) && same_device(topk_ids) &&
           valid_optional(expert_map) && valid_optional(global_scale1) &&
           valid_optional(global_scale2) && valid_optional(g_idx1) &&
           valid_optional(g_idx2) && valid_optional(sort_indices1) &&
           valid_optional(sort_indices2) && valid_optional(w1_zeros) &&
           valid_optional(w2_zeros) && valid_optional(workspace) &&
           same_device(out) &&
           "`FusedMarlinMoe` requires all tensors on the input device");
    if (workspace) {
      assert(workspace->ndim() == 1 && workspace->numel() > 0 &&
             workspace->dtype() == DataType::kInt32 &&
             "`FusedMarlinMoe` requires a non-empty int32 `workspace`");
    }

    assert(out.shape() == hidden_states.shape() &&
           out.dtype() == hidden_states.dtype() && out.IsContiguous() &&
           "`FusedMarlinMoe` output metadata must match `hidden_states`");
    const auto aliases = hidden_states.data() == out.data();
    assert(aliases == inplace_ &&
           "`FusedMarlinMoe` `out` must alias `hidden_states` exactly when "
           "`inplace` is true");

    assert(num_tokens <= std::numeric_limits<Tensor::Size>::max() / topk &&
           hidden_size <= std::numeric_limits<Tensor::Size>::max() / 16 &&
           "`FusedMarlinMoe` dimensions overflow");
    [[maybe_unused]] const auto route_count = num_tokens * topk;
    assert(route_count <= std::numeric_limits<int32_t>::max() &&
           routed_num_experts <=
               (std::numeric_limits<int32_t>::max() - route_count) / 63 &&
           "`FusedMarlinMoe` routing dimensions overflow int32 indices");
  }

  Tensor hidden_states_metadata_;

  Tensor w1_metadata_;

  Tensor w2_metadata_;

  Tensor w1_scale_metadata_;

  Tensor w2_scale_metadata_;

  Tensor gating_output_metadata_;

  Tensor topk_weights_metadata_;

  Tensor topk_ids_metadata_;

  std::optional<Tensor> expert_map_metadata_;

  std::optional<Tensor> global_scale1_metadata_;

  std::optional<Tensor> global_scale2_metadata_;

  std::optional<Tensor> g_idx1_metadata_;

  std::optional<Tensor> g_idx2_metadata_;

  std::optional<Tensor> sort_indices1_metadata_;

  std::optional<Tensor> sort_indices2_metadata_;

  std::optional<Tensor> w1_zeros_metadata_;

  std::optional<Tensor> w2_zeros_metadata_;

  std::optional<Tensor> workspace_metadata_;

  Tensor out_metadata_;

  int64_t quant_type_id_{0};

  bool apply_router_weight_on_input_{false};

  int64_t global_num_experts_{-1};

  bool is_k_full_{true};

  bool inplace_{false};

 protected:
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_FUSED_MARLIN_MOE_H_
