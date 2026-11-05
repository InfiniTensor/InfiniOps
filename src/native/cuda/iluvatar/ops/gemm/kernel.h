#ifndef INFINI_OPS_ILUVATAR_GEMM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_GEMM_KERNEL_H_

#include <cstddef>

#include "native/cuda/iluvatar/ops/gemm/kernel.cuh"

namespace infini::ops::iluvatar {

template <int kInput, int kOutput>
inline void LaunchSmallBatchGemv(const void* a, const void* b, void* y,
                                 std::size_t batch, cudaStream_t stream) {
  const auto* x = static_cast<const __nv_bfloat16*>(a);
  const auto* w = static_cast<const __nv_bfloat16*>(b);
  auto* out = static_cast<__nv_bfloat16*>(y);
  if (batch == 1) {
    SmallBatchGemvKernel<1, kInput, kOutput>
        <<<kOutput / 4, 128, 0, stream>>>(x, w, out);
  } else {
    SmallBatchGemvKernel<2, kInput, kOutput>
        <<<kOutput / 4, 128, 0, stream>>>(x, w, out);
  }
}

}  // namespace infini::ops::iluvatar

#endif
