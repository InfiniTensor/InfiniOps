#ifndef INFINI_OPS_CUDA_CLOVER_MASK_ROUTES_KERNEL_CUH_
#define INFINI_OPS_CUDA_CLOVER_MASK_ROUTES_KERNEL_CUH_

#include <cstdint>

namespace infini::ops {

// Shared-layer SIMT kernel for CloverMaskRoutes, mirroring stage 1 of the
// Triton `clover_live_rows` kernel: live rows pass their route through
// unchanged (negative/dead ids included), non-live rows are replaced by
// `num_experts` (the routing mask value). `ids` layout is [rows, topk] and
// `live` is per row.
template <Device::Type kDevice, typename Index>
__global__ void CloverMaskRoutesKernel(const Index* __restrict__ ids,
                                       const int8_t* __restrict__ live,
                                       int64_t num_experts, int64_t width,
                                       Index* __restrict__ out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    static_cast<int64_t>(threadIdx.x);
  if (i >= n) return;
  const int64_t row = i / width;
  out[i] = live[row] != 0 ? ids[i] : static_cast<Index>(num_experts);
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_MASK_ROUTES_KERNEL_CUH_
