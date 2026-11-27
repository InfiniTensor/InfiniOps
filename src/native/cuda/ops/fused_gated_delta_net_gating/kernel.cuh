#ifndef INFINI_OPS_CUDA_FUSED_GATED_DELTA_NET_GATING_KERNEL_CUH_
#define INFINI_OPS_CUDA_FUSED_GATED_DELTA_NET_GATING_KERNEL_CUH_

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "data_type.h"

namespace infini::ops {
namespace gating_detail {

struct Layout {
  ptrdiff_t strides[6][3]{};
  size_t total, seq, hidden;
  float beta, threshold;
};

// One thread per (batch, sequence, hidden) element turns the gating
// projections into the log decay `g` and the delta-rule coefficient `betao`.
template <Device::Type kDevice, typename T>
__global__ void GatingKernel(const T* A, const T* a, const T* b, const T* dt,
                             float* g, float* betao, Layout p) {
  const size_t linear =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= p.total) return;
  const size_t h = linear % p.hidden, s = (linear / p.hidden) % p.seq;
  const size_t batch = linear / p.hidden / p.seq;
  ptrdiff_t offsets[6];
  for (int i = 0; i < 6; ++i)
    offsets[i] = static_cast<ptrdiff_t>(batch) * p.strides[i][0] +
                 static_cast<ptrdiff_t>(s) * p.strides[i][1] +
                 static_cast<ptrdiff_t>(h) * p.strides[i][2];
  const float x = static_cast<float>(a[offsets[1]]) +
                  static_cast<float>(dt[h * p.strides[3][0]]);
  const float bx = p.beta * x;
  const float softplus = bx <= p.threshold ? log1pf(expf(bx)) / p.beta : x;
  g[offsets[4]] = -expf(static_cast<float>(A[h * p.strides[0][0]])) * softplus;
  const float bv = static_cast<float>(b[offsets[2]]);
  const float z = expf(bv >= 0 ? -bv : bv);
  betao[offsets[5]] = bv >= 0 ? 1.0f / (1.0f + z) : z / (1.0f + z);
}

}  // namespace gating_detail
}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_FUSED_GATED_DELTA_NET_GATING_KERNEL_CUH_
