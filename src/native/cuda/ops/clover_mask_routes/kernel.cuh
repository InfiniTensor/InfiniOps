#ifndef INFINI_OPS_CUDA_CLOVER_MASK_ROUTES_KERNEL_CUH_
#define INFINI_OPS_CUDA_CLOVER_MASK_ROUTES_KERNEL_CUH_

#include <cstdint>

#include "data_type.h"

namespace infini::ops {

// Shared-layer SIMT kernel for CloverMaskRoutes, mirroring stage 1 of the
// Triton `clover_live_rows` kernel: live rows pass their route through
// unchanged (negative/dead ids included), non-live rows are replaced by
// `num_experts` (the routing mask value). `ids` layout is [rows, topk] and
// `live` is per row.
//
// A row's width that is a power of two (the usual top-k) makes the row a
// shift of the element index; any other width takes the kernel that divides.
namespace clover_mask_routes_detail {

template <typename Index>
__device__ __forceinline__ void MaskRoute(const Index* __restrict__ ids,
                                          const int8_t* __restrict__ live,
                                          Index mask, int64_t row, int64_t i,
                                          Index* __restrict__ out) {
  // Both are read before the flag selects: a load that waits for the flag
  // would cost a second memory latency.
  const Index route = ids[i];
  const int8_t flag = live[row];
  out[i] = flag != 0 ? route : mask;
}

}  // namespace clover_mask_routes_detail

template <Device::Type kDevice, typename Index>
__global__ void CloverMaskRoutesKernel(const Index* __restrict__ ids,
                                       const int8_t* __restrict__ live,
                                       Index mask, int shift,
                                       Index* __restrict__ out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    static_cast<int64_t>(threadIdx.x);
  if (i >= n) return;
  clover_mask_routes_detail::MaskRoute(ids, live, mask, i >> shift, i, out);
}

template <Device::Type kDevice, typename Index>
__global__ void CloverMaskRoutesByDivisionKernel(
    const Index* __restrict__ ids, const int8_t* __restrict__ live, Index mask,
    int64_t width, Index* __restrict__ out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    static_cast<int64_t>(threadIdx.x);
  if (i >= n) return;
  clover_mask_routes_detail::MaskRoute(ids, live, mask, i / width, i, out);
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_MASK_ROUTES_KERNEL_CUH_
