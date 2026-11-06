#ifndef INFINI_OPS_CUDA_PAGED_ATTENTION_REDUCTION_POLICY_H_
#define INFINI_OPS_CUDA_PAGED_ATTENTION_REDUCTION_POLICY_H_
#include <cstddef>
namespace infini::ops::detail {
inline bool CanBatchDecodeQkReduction(
    bool iluvatar, bool bf16, bool int32, std::size_t batch, std::size_t heads,
    std::size_t kv, std::size_t dim, std::size_t blocks, std::size_t block,
    int splits, bool alibi, std::ptrdiff_t qs, std::ptrdiff_t qh,
    std::ptrdiff_t os, std::ptrdiff_t oh, std::ptrdiff_t kb, std::ptrdiff_t ks,
    std::ptrdiff_t kh, std::ptrdiff_t vb, std::ptrdiff_t vs,
    std::ptrdiff_t vh) {
  return iluvatar && bf16 && int32 && (batch == 1 || batch == 2) &&
         heads == 32 && kv == 2 && dim == 128 && blocks == 1 && block == 4096 &&
         splits == 4 && !alibi && qs == 4096 && qh == 128 && os == 4096 &&
         oh == 128 && kb == 1048576 && ks == 128 && kh == 524288 &&
         vb == 1048576 && vs == 128 && vh == 524288;
}
}  // namespace infini::ops::detail

#endif
