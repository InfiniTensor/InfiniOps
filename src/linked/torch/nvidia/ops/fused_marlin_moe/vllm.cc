#include "linked/torch/nvidia/ops/fused_marlin_moe/vllm.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <cuda_runtime_api.h>

#include <utility>

namespace infini::ops::linked::torch::nvidia {

void VllmFusedMarlinMoe::Validate(const int64_t quant_type_id,
                                  const bool has_global_scale,
                                  const bool has_w1_zeros,
                                  const bool has_w2_zeros) {
  VllmMoeWna16MarlinGemm::Validate(quant_type_id, has_global_scale);

  constexpr int64_t kUint4 = 1125899906843648;
  constexpr int64_t kUint4B8 = 1125899907892224;
  constexpr int64_t kUint8B128 = 1125899923621888;
  constexpr int64_t kFloat8E4M3Fn = 2814749767172868;
  TORCH_CHECK(quant_type_id == kUint4 || quant_type_id == kUint4B8 ||
                  quant_type_id == kUint8B128 || quant_type_id == kFloat8E4M3Fn,
              "Linked `fused_marlin_moe` received an unsupported "
              "`quant_type_id`.");

  const auto expects_zeros = quant_type_id == kUint4;
  TORCH_CHECK(has_w1_zeros == expects_zeros && has_w2_zeros == expects_zeros,
              "Linked `fused_marlin_moe` requires zero points for both weight "
              "tensors exactly when `quant_type_id` is `uint4`.");
}

int64_t VllmFusedMarlinMoe::WorkspaceSize(const int device_index) {
  cudaDeviceProp properties{};
  const auto status = cudaGetDeviceProperties(&properties, device_index);
  TORCH_CHECK(status == cudaSuccess,
              "Failed to query the CUDA device for linked "
              "`fused_marlin_moe`.");

  return static_cast<int64_t>(properties.multiProcessorCount) * 4;
}

bool VllmFusedMarlinMoe::UseAtomicAdd(const at::ScalarType dtype,
                                      const int device_index) {
  cudaDeviceProp properties{};
  const auto status = cudaGetDeviceProperties(&properties, device_index);
  TORCH_CHECK(status == cudaSuccess,
              "Failed to query the CUDA device for linked "
              "`fused_marlin_moe`.");

  return dtype == at::kHalf || properties.major >= 9;
}

void VllmFusedMarlinMoe::CallAlign(at::Tensor topk_ids,
                                   const int64_t num_experts,
                                   const int64_t block_size,
                                   at::Tensor sorted_token_ids,
                                   at::Tensor expert_ids,
                                   at::Tensor num_tokens_post_padded) {
  static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_moe_C::moe_align_block_size", "");
  c10::Stack stack;
  stack.reserve(6);
  stack.emplace_back(std::move(topk_ids));
  stack.emplace_back(num_experts);
  stack.emplace_back(block_size);
  stack.emplace_back(std::move(sorted_token_ids));
  stack.emplace_back(std::move(expert_ids));
  stack.emplace_back(std::move(num_tokens_post_padded));
  op.callBoxed(&stack);

  TORCH_CHECK(stack.empty(),
              "Linked `moe_align_block_size` returned an unexpected value.");
}

void VllmFusedMarlinMoe::CallMarlin(
    at::Tensor a, at::Tensor out, at::Tensor b_q_weight, at::Tensor b_scales,
    std::optional<at::Tensor> global_scale,
    std::optional<at::Tensor> b_zeros_or_none,
    std::optional<at::Tensor> g_idx_or_none,
    std::optional<at::Tensor> perm_or_none, at::Tensor workspace,
    at::Tensor sorted_token_ids, at::Tensor expert_ids,
    at::Tensor num_tokens_past_padded, at::Tensor topk_weights,
    const int64_t moe_block_size, const int64_t top_k,
    const bool mul_topk_weights, const bool is_ep, const int64_t b_q_type_id,
    const int64_t size_m, const int64_t size_n, const int64_t size_k,
    const bool is_full_k, const bool use_atomic_add, const bool use_fp32_reduce,
    const bool is_zp_float) {
  VllmMoeWna16MarlinGemm::Call(
      std::move(a), std::move(out), std::move(b_q_weight), std::move(b_scales),
      std::move(global_scale), std::move(b_zeros_or_none),
      std::move(g_idx_or_none), std::move(perm_or_none), std::move(workspace),
      std::move(sorted_token_ids), std::move(expert_ids),
      std::move(num_tokens_past_padded), std::move(topk_weights),
      moe_block_size, top_k, mul_topk_weights, is_ep, b_q_type_id, size_m,
      size_n, size_k, is_full_k, use_atomic_add, use_fp32_reduce, is_zp_float);
}

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

template class TorchFusedMarlinMoe<
    ::infini::ops::linked::torch::nvidia::VllmFusedMarlinMoe>;

}  // namespace infini::ops::linked::torch
