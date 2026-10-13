#ifndef INFINI_OPS_QY_ROPE_PACK_KERNEL_H_
#define INFINI_OPS_QY_ROPE_PACK_KERNEL_H_

#include <cstddef>
#include <cstdint>

namespace infini::ops::qy {

void rope_pack_launch(void *dst, const void *src, size_t seq, size_t heads,
                      size_t dim, int64_t src_stride0, int64_t src_stride1,
                      int64_t src_stride2, int elem_bytes, void *stream);

void rope_unpack_launch(void *dst, const void *src, size_t seq, size_t heads,
                        size_t dim, int64_t dst_stride0, int64_t dst_stride1,
                        int64_t dst_stride2, int elem_bytes, void *stream);

}  // namespace infini::ops::qy

#endif  // INFINI_OPS_QY_ROPE_PACK_KERNEL_H_
