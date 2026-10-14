#ifndef INFINI_OPS_MOE_TOPK_SIGMOID_KERNEL_H_
#define INFINI_OPS_MOE_TOPK_SIGMOID_KERNEL_H_

#include <cstddef>

namespace infini::ops::qy {

// Contiguous MoE top-k after sigmoid (no CUB).
// gating: [tokens, experts] (f16/bf16/f32)
// correction_bias: [experts] (f16/bf16/f32) or null; bias_dtype_tag ignored if
// null topk_weights: [tokens, topk] (f16/bf16/f32); topk_indices: [tokens,
// topk] (i32/i64) dtype tags match infini::rt::DataType.
void moe_topk_sigmoid_launch(void *topk_weights, void *topk_indices,
                             const void *gating, const void *correction_bias,
                             size_t tokens, size_t experts, size_t topk,
                             bool renormalize, int gating_dtype_tag,
                             int weight_dtype_tag, int index_dtype_tag,
                             int bias_dtype_tag, void *stream);

}  // namespace infini::ops::qy

#endif  // INFINI_OPS_MOE_TOPK_SIGMOID_KERNEL_H_
