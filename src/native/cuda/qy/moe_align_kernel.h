#ifndef INFINI_OPS_QY_MOE_ALIGN_KERNEL_H_
#define INFINI_OPS_QY_MOE_ALIGN_KERNEL_H_

#include <cstddef>
#include <cstdint>

namespace infini::ops::qy {

// Workspace bytes for moe_align_launch: 2 * (num_experts + 1) * sizeof(int32).
inline size_t moe_align_workspace_bytes(size_t num_experts) {
  return 2 * (num_experts + 1) * sizeof(int32_t);
}

// Host-free contiguous moe_align (Denglin-safe block/grid caps).
// topk_ids / expert_map: int32; sorted_token_ids / expert_ids /
// num_tokens_post_padded: int32. expert_map may be nullptr.
void moe_align_launch(void *sorted_token_ids, void *expert_ids,
                      void *num_tokens_post_padded, const void *topk_ids,
                      const void *expert_map, size_t numel, size_t num_experts,
                      size_t block_size, size_t sorted_capacity,
                      size_t expert_ids_capacity, bool pad_sorted_token_ids,
                      void *workspace, void *stream);

}  // namespace infini::ops::qy

#endif  // INFINI_OPS_QY_MOE_ALIGN_KERNEL_H_
