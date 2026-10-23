#ifndef INFINI_OPS_ROPE_PACK_KERNEL_H_
#define INFINI_OPS_ROPE_PACK_KERNEL_H_

#include <cstddef>
#include <cstdint>

namespace infini::ops::qy {

// Pack strided [seq, heads, dim] (elem_bytes 2 or 4) into contiguous row-major.
// src_stride{0,1,2} are in elements. dst is contiguous [seq, heads, dim].
void rope_pack_launch(void *dst, const void *src, size_t seq, size_t heads,
                      size_t dim, int64_t src_stride0, int64_t src_stride1,
                      int64_t src_stride2, int elem_bytes, void *stream);

// Unpack contiguous [seq, heads, dim] into strided dst with the given strides.
void rope_unpack_launch(void *dst, const void *src, size_t seq, size_t heads,
                        size_t dim, int64_t dst_stride0, int64_t dst_stride1,
                        int64_t dst_stride2, int elem_bytes, void *stream);

}  // namespace infini::ops::qy

#endif  // INFINI_OPS_ROPE_PACK_KERNEL_H_
