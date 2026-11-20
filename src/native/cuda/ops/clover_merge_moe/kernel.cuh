#ifndef INFINI_OPS_CUDA_CLOVER_MERGE_MOE_KERNEL_CUH_
#define INFINI_OPS_CUDA_CLOVER_MERGE_MOE_KERNEL_CUH_

#include <cstdint>

namespace infini::ops {

// Shared-layer SIMT kernel for CloverMergeMoe, mirroring stage 2 of the
// Triton `clover_live_rows` kernel: freshly computed values replace the
// cache on live rows and are propagated to the output; cached values are
// reused on non-live rows. `fresh`/`cache`/`out` share [rows, width].
template <Device::Type kDevice, typename Data>
__global__ void CloverMergeMoeKernel(const Data* __restrict__ fresh,
                                     const int8_t* __restrict__ live,
                                     Data* __restrict__ cache, Data* __restrict__ out,
                                     int64_t width, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    static_cast<int64_t>(threadIdx.x);
  if (i >= n) return;
  const int64_t row = i / width;
  const Data value = live[row] != 0 ? fresh[i] : cache[i];
  out[i] = value;
  cache[i] = value;
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_MERGE_MOE_KERNEL_CUH_
