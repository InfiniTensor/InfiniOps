#ifndef INFINI_OPS_CUDA_CLOVER_CORESET_KERNEL_CUH_
#define INFINI_OPS_CUDA_CLOVER_CORESET_KERNEL_CUH_

#include <cmath>
#include <cstdint>
#include <limits>

namespace infini::ops {

// Shared-layer SIMT kernels for CloverCoreset, mirroring the three Triton
// stages: (0) per block, accumulate routing mass per expert over the union
// of each row's top-8 experts; (1) per block, keep experts by descending
// mass until the cumulative NORMALIZED probability reaches top_p (mass is
// normalized by the block total, matching the Triton reference); (2) per
// token, apply the block's allow-list to its logits row. Layout [tokens,256].

constexpr int kCloverCoresetExperts = 256;

template <Device::Type kDevice, typename Data>
__global__ void CloverCoresetMassKernel(const Data* __restrict__ logits,
                                        float* __restrict__ mass,
                                        int width, int64_t blocks) {
  // One thread per expert (blockDim = 256), iterating the block's rows
  // serially. Deterministic by construction: no atomics, fixed order.
  const int64_t block = static_cast<int64_t>(blockIdx.x);
  if (block >= blocks) return;
  const int expert = static_cast<int>(threadIdx.x);  // 0..255

  float total = 0.0f;
  // A row counts this expert only if the expert is in that row's top-8
  // (ties resolved by ascending expert id, like the Triton tie_break_left).
  for (int t = 0; t < width; ++t) {
    const Data* row =
        logits + (block * width + t) * kCloverCoresetExperts;
    const float my_logit = static_cast<float>(row[expert]);
    // Rank of my expert in this row: count strictly-greater logits plus
    // equal logits with a smaller id.
    int stronger = 0;
    float row_max = -INFINITY;
    float denom = 0.0f;
    for (int e = 0; e < kCloverCoresetExperts; ++e) {
      const float v = static_cast<float>(row[e]);
      if (v > row_max) row_max = v;
    }
    for (int e = 0; e < kCloverCoresetExperts; ++e) {
      const float v = static_cast<float>(row[e]);
      denom += expf(v - row_max);
      if (v > my_logit || (v == my_logit && e < expert)) ++stronger;
    }
    if (stronger < 8) {
      total += expf(my_logit - row_max) / denom;
    }
  }
  mass[block * kCloverCoresetExperts + expert] = total;
}

template <Device::Type kDevice>
__global__ void CloverCoresetAllowKernel(const float* __restrict__ mass,
                                         int8_t* __restrict__ allow,
                                         int64_t blocks, float top_p) {
  const int64_t block = static_cast<int64_t>(blockIdx.x);
  if (block >= blocks) return;
  const float* block_mass = mass + block * kCloverCoresetExperts;

  extern __shared__ float sorted[];  // [256]
  __shared__ float total[1];
  __shared__ int allow_shared[kCloverCoresetExperts];
  for (int e = threadIdx.x; e < kCloverCoresetExperts; e += blockDim.x)
    sorted[e] = block_mass[e];
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (int e = 0; e < kCloverCoresetExperts; ++e) sum += sorted[e];
    total[0] = sum > 1.0e-9f ? sum : 1.0e-9f;
  }
  for (int e = threadIdx.x; e < kCloverCoresetExperts; e += blockDim.x)
    allow_shared[e] = 0;
  __syncthreads();

  // Serial selection by thread 0, matching the Triton reference: keep
  // experts in descending-mass order while the cumulative probability
  // (mass normalized by the block total) BEFORE each element stays below
  // top_p — i.e., include the first element that reaches the threshold.
  if (threadIdx.x == 0) {
    const bool keep_positive_only = top_p >= 1.0f;
    float kept = 0.0f;
    for (int picked = 0; picked < kCloverCoresetExperts; ++picked) {
      int best = -1;
      float best_v = 0.0f;  // ties keep expert id ascending, like Triton's key
      for (int e = 0; e < kCloverCoresetExperts; ++e) {
        if (sorted[e] > best_v) {
          best_v = sorted[e];
          best = e;
        }
      }
      if (best < 0) break;
      sorted[best] = -1.0f;
      if (keep_positive_only) {
        allow_shared[best] = 1;  // top_p >= 1: keep every positive-mass expert
      } else {
        allow_shared[best] = 1;
        kept += best_v / total[0];
        if (kept >= top_p) break;
      }
    }
    // The Triton reference ORs `keep` with (position == 0) in the
    // sorted coordinate system, which keeps the highest-mass expert —
    // already guaranteed by the first pick above. No extra action.
  }
  __syncthreads();

  for (int e = threadIdx.x; e < kCloverCoresetExperts; e += blockDim.x)
    allow[block * kCloverCoresetExperts + e] =
        static_cast<int8_t>(allow_shared[e]);
}

template <Device::Type kDevice, typename Data>
__global__ void CloverCoresetApplyKernel(const Data* __restrict__ logits,
                                         const int8_t* __restrict__ allow,
                                         Data* __restrict__ out, int width,
                                         int64_t tokens) {
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  if (token >= tokens) return;
  const int64_t block = token / width;
  const Data* row = logits + token * kCloverCoresetExperts;
  Data* dst = out + token * kCloverCoresetExperts;
  const int8_t* block_allow = allow + block * kCloverCoresetExperts;
  for (int e = threadIdx.x; e < kCloverCoresetExperts; e += blockDim.x)
    dst[e] = block_allow[e] != 0
                 ? row[e]
                 : static_cast<Data>(-INFINITY);
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_CORESET_KERNEL_CUH_
