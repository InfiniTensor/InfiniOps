#ifndef INFINI_OPS_ILUVATAR_GEMM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_GEMM_KERNEL_H_

#include <cstddef>

#include "native/cuda/iluvatar/ops/gemm/kernel.cuh"

namespace infini::ops::iluvatar {

template <int kBatch>
inline void LaunchSmallBatchGemv(const __nv_bfloat16* x, const __nv_bfloat16* w,
                                 __nv_bfloat16* out, std::size_t input,
                                 std::size_t output, cudaStream_t stream) {
  if (input % 256 == 0) {
    SmallBatchGemvKernel<kBatch, false>
        <<<output / 4, 128, 0, stream>>>(x, w, out, input, output);
  } else {
    SmallBatchGemvKernel<kBatch, true>
        <<<output / 4, 128, 0, stream>>>(x, w, out, input, output);
  }
}

inline void LaunchSmallBatchGemv(const void* a, const void* b, void* y,
                                 std::size_t batch, std::size_t input,
                                 std::size_t output, cudaStream_t stream) {
  const auto* x = static_cast<const __nv_bfloat16*>(a);
  const auto* w = static_cast<const __nv_bfloat16*>(b);
  auto* out = static_cast<__nv_bfloat16*>(y);
  if (batch == 1) {
    LaunchSmallBatchGemv<1>(x, w, out, input, output, stream);
  } else {
    LaunchSmallBatchGemv<2>(x, w, out, input, output, stream);
  }
}

}  // namespace infini::ops::iluvatar

#endif
