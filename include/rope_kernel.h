#ifndef INFINI_OPS_ROPE_KERNEL_H_
#define INFINI_OPS_ROPE_KERNEL_H_

#include <cstddef>
#include <cstdint>

namespace infini::ops::qy {

// Contiguous packed RoPE on [seqlen, nhead, head_dim] (nhead may be 1).
// pos_ids: int64 [seqlen]; sin/cos tables: [max_pos, head_dim/2], same dtype as
// x. algo_tag: 0 = GPT_J (interleaved), 1 = GPT_NEOX (half-split). dtype_tag
// matches infini::rt::DataType (kFloat16 / kBFloat16 / kFloat32).
void rope_launch(void *out, const void *in, const int64_t *pos_ids,
                 const void *sin_table, const void *cos_table, size_t seqlen,
                 size_t nhead, size_t head_dim, int algo_tag, int dtype_tag,
                 void *stream);

}  // namespace infini::ops::qy

#endif  // INFINI_OPS_ROPE_KERNEL_H_
