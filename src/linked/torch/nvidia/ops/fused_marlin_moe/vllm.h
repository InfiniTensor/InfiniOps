#ifndef INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_FUSED_MARLIN_MOE_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_FUSED_MARLIN_MOE_VLLM_H_

#include <optional>

#include "linked/torch/nvidia/ops/moe_wna16_marlin_gemm/vllm.h"
#include "linked/torch/ops/fused_marlin_moe.h"

namespace infini::ops::linked::torch::nvidia {

struct VllmFusedMarlinMoe : C10<Device::Type::kNvidia> {
  static void Validate(int64_t quant_type_id, bool has_global_scale,
                       bool has_w1_zeros, bool has_w2_zeros);

  static int64_t WorkspaceSize(int device_index);

  static bool UseAtomicAdd(at::ScalarType dtype, int device_index);

  static void CallAlign(at::Tensor topk_ids, int64_t num_experts,
                        int64_t block_size, at::Tensor sorted_token_ids,
                        at::Tensor expert_ids,
                        at::Tensor num_tokens_post_padded);

  static void CallMarlin(at::Tensor a, at::Tensor out, at::Tensor b_q_weight,
                         at::Tensor b_scales,
                         std::optional<at::Tensor> global_scale,
                         std::optional<at::Tensor> b_zeros_or_none,
                         std::optional<at::Tensor> g_idx_or_none,
                         std::optional<at::Tensor> perm_or_none,
                         at::Tensor workspace, at::Tensor sorted_token_ids,
                         at::Tensor expert_ids,
                         at::Tensor num_tokens_past_padded,
                         at::Tensor topk_weights, int64_t moe_block_size,
                         int64_t top_k, bool mul_topk_weights, bool is_ep,
                         int64_t b_q_type_id, int64_t size_m, int64_t size_n,
                         int64_t size_k, bool is_full_k, bool use_atomic_add,
                         bool use_fp32_reduce, bool is_zp_float);
};

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

extern template class TorchFusedMarlinMoe<
    ::infini::ops::linked::torch::nvidia::VllmFusedMarlinMoe>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<FusedMarlinMoe, Device::Type::kNvidia, 16>
    : public linked::torch::TorchFusedMarlinMoe<
          linked::torch::nvidia::VllmFusedMarlinMoe> {
 public:
  using linked::torch::TorchFusedMarlinMoe<
      linked::torch::nvidia::VllmFusedMarlinMoe>::TorchFusedMarlinMoe;

  using linked::torch::TorchFusedMarlinMoe<
      linked::torch::nvidia::VllmFusedMarlinMoe>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_FUSED_MARLIN_MOE_VLLM_H_
