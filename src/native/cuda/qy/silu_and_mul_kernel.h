#ifndef INFINI_OPS_QY_SILU_AND_MUL_KERNEL_H_
#define INFINI_OPS_QY_SILU_AND_MUL_KERNEL_H_

#include <cstddef>

namespace infini::ops::qy {

// Contiguous gated SiLU: out[r, c] = silu(in[r, c]) * in[r, c + hidden]
// in is [rows, 2*hidden], out is [rows, hidden].
// dtype_tag matches infini::rt::DataType (kFloat16 / kBFloat16 / kFloat32).
void silu_and_mul_launch(void *out, const void *in, size_t rows, size_t hidden,
                         int dtype_tag, void *stream);

}  // namespace infini::ops::qy

#endif  // INFINI_OPS_QY_SILU_AND_MUL_KERNEL_H_
