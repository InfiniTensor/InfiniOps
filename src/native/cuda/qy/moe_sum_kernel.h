#ifndef INFINI_OPS_QY_MOE_SUM_KERNEL_H_
#define INFINI_OPS_QY_MOE_SUM_KERNEL_H_

#include <cstddef>

namespace infini::ops::qy {

// Contiguous expert combine: out[t, h] = sum_k in[t, k, h]
// in is [tokens, topk, hidden], out is [tokens, hidden].
// dtype_tag matches infini::rt::DataType (kFloat16 / kBFloat16 / kFloat32).
void moe_sum_launch(void *out, const void *in, size_t tokens, size_t topk,
                    size_t hidden, int dtype_tag, void *stream);

}  // namespace infini::ops::qy

#endif  // INFINI_OPS_QY_MOE_SUM_KERNEL_H_
