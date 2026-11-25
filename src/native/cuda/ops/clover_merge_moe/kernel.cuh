#ifndef INFINI_OPS_CUDA_CLOVER_MERGE_MOE_KERNEL_CUH_
#define INFINI_OPS_CUDA_CLOVER_MERGE_MOE_KERNEL_CUH_

#include <cstdint>

#include "data_type.h"

namespace infini::ops {

// Shared-layer SIMT kernel for CloverMergeMoe, mirroring stage 2 of the
// Triton `clover_live_rows` kernel: freshly computed values replace the
// cache on live rows and are propagated to the output; cached values are
// reused on non-live rows. `fresh`/`cache`/`out` share [rows, width].
//
// A thread block covers columns of one row, so the row's flag is a single
// load for all of its threads, and no thread has to divide to find its row.
// Only a live row's cache entry is written back: a dead row's already holds
// the value it keeps.
//
// Grid: x = rows (the grid's x axis takes far more than the 65535 of y
// and z), y = the row's columns, one element or one 16-byte vector per
// thread.

template <Device::Type kDevice, typename Data>
__global__ void CloverMergeMoeKernel(const Data* __restrict__ fresh,
                                     const int8_t* __restrict__ live,
                                     Data* __restrict__ cache,
                                     Data* __restrict__ out,
                                     int64_t width) {
  const int64_t column =
      static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (column >= width) return;
  const int64_t row = blockIdx.x;
  const int64_t i = row * width + column;
  const bool on = live[row] != 0;
  const Data value = on ? fresh[i] : cache[i];
  out[i] = value;
  if (on) cache[i] = value;
}

// The same with 16 bytes per thread: `vectors` is a row's width in such
// vectors and the three tensors start on a 16-byte boundary.
template <Device::Type kDevice, typename Data>
__global__ void CloverMergeMoeVectorKernel(
    const Data* __restrict__ fresh, const int8_t* __restrict__ live,
    Data* __restrict__ cache, Data* __restrict__ out, int64_t vectors) {
  const int64_t column =
      static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (column >= vectors) return;
  const int64_t row = blockIdx.x;
  const int64_t i = row * vectors + column;
  const bool on = live[row] != 0;
  const uint4 value = on ? reinterpret_cast<const uint4*>(fresh)[i]
                         : reinterpret_cast<const uint4*>(cache)[i];
  reinterpret_cast<uint4*>(out)[i] = value;
  if (on) reinterpret_cast<uint4*>(cache)[i] = value;
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_MERGE_MOE_KERNEL_CUH_
