#ifndef INFINI_OPS_CUDA_CLOVER_LIVE_MASK_KERNEL_CUH_
#define INFINI_OPS_CUDA_CLOVER_LIVE_MASK_KERNEL_CUH_

#include <cstdint>

namespace infini::ops {

// Shared-layer SIMT kernel for CloverLiveMask, mirroring stage 0 of the
// Triton `clover_live_rows` kernel: a row is live when it is a mask now, was
// a mask in the previous snapshot, or the caller forces a refresh. The
// current ids are written back into the snapshot for the next call.
template <Device::Type kDevice, typename Index>
__global__ void CloverLiveMaskKernel(const Index* __restrict__ ids,
                                     Index* __restrict__ previous,
                                     int64_t mask_id, int refresh,
                                     int8_t* __restrict__ live, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    static_cast<int64_t>(threadIdx.x);
  if (i >= n) return;
  const Index current = ids[i];
  const Index prev = refresh ? static_cast<Index>(mask_id) : previous[i];
  live[i] = (current == static_cast<Index>(mask_id)) ||
                    (prev == static_cast<Index>(mask_id)) || refresh
                ? 1
                : 0;
  previous[i] = current;
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_LIVE_MASK_KERNEL_CUH_
