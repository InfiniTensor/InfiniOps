#ifndef INFINI_OPS_KUNLUN_ROTARY_EMBEDDING_LAUNCH_H_
#define INFINI_OPS_KUNLUN_ROTARY_EMBEDDING_LAUNCH_H_

#include <xpu/runtime.h>

#include <cstddef>
#include <cstdint>

namespace infini::ops {

template <typename T, typename TCache>
void LaunchRotaryEmbeddingKunlun(XPUStream queue, int cluster_count,
                                 const int64_t* positions, T* data,
                                 const TCache* cos_sin_cache, size_t cache_rows,
                                 size_t num_tokens, size_t sequence_length,
                                 size_t num_heads, size_t rot_dim,
                                 size_t rope_dim_offset, ptrdiff_t batch_stride,
                                 ptrdiff_t token_stride, ptrdiff_t head_stride,
                                 ptrdiff_t cache_token_stride, bool is_neox,
                                 bool inverse);

}  // namespace infini::ops

#endif
