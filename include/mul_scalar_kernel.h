#ifndef INFINI_OPS_MUL_SCALAR_KERNEL_H_
#define INFINI_OPS_MUL_SCALAR_KERNEL_H_

#include <cstddef>

namespace infini::ops::qy {

// Contiguous elementwise out[i] = in[i] * alpha.
// dtype_tag matches infini::rt::DataType (kFloat16 / kBFloat16 / kFloat32 /
// kFloat64).
void mul_scalar_launch(void *out, const void *in, size_t numel, int dtype_tag,
                       double alpha, void *stream);

}  // namespace infini::ops::qy

#endif  // INFINI_OPS_MUL_SCALAR_KERNEL_H_
