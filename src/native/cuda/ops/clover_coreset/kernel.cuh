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

// Stage A: one thread block (256 threads) per routing row. Finds the
// row's top-8 experts with 8 fixed-order selection rounds and writes
// (expert_id, normalized_probability) into a per-row staging buffer.
template <Device::Type kDevice, typename Data>
__global__ void CloverCoresetRowKernel(const Data* __restrict__ logits,
                                       float* __restrict__ row_top8,
                                       int width, int64_t tokens) {
  constexpr int kExperts = kCloverCoresetExperts;  // 256
  constexpr int kWarps = kExperts / 32;

  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= tokens) return;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % 32;
  const int warp = tid / 32;

  extern __shared__ float smem[];
  float* row_f32 = smem;                   // [256]
  float* red = smem + kExperts;            // [8]
  int* red_id = reinterpret_cast<int*>(smem + kExperts + kWarps);
  int* taken = reinterpret_cast<int*>(smem + kExperts + 2 * kWarps);

  const Data* row_ptr = logits + row * kExperts;
  row_f32[tid] = static_cast<float>(row_ptr[tid]);
  taken[tid] = 0;
  __syncthreads();

  float m = row_f32[tid];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  if (lane == 0) red[warp] = m;
  __syncthreads();
  if (tid == 0) {
    float mm = red[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) mm = fmaxf(mm, red[w]);
    red[0] = mm;
  }
  __syncthreads();
  const float row_max = red[0];

  float d = expf(row_f32[tid] - row_max);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    d += __shfl_xor_sync(0xffffffffu, d, off);
  if (lane == 0) red[warp] = d;
  __syncthreads();
  if (tid == 0) {
    float dd = red[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) dd += red[w];
    red[0] = dd;
  }
  __syncthreads();
  const float denom = red[0];

  // Top-8 by successive block argmax over untaken experts (ties prefer
  // the smaller id, matching the Triton bitcast key).
#pragma unroll
  for (int rank = 0; rank < 8; ++rank) {
    float v = taken[tid] ? -INFINITY : row_f32[tid];
    int id = tid;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, v, off);
      const int oid = __shfl_xor_sync(0xffffffffu, id, off);
      if (ov > v || (ov == v && oid < id)) { v = ov; id = oid; }
    }
    if (lane == 0) { red[warp] = v; red_id[warp] = id; }
    __syncthreads();
    if (tid == 0) {
      float bv = red[0]; int bi = red_id[0];
#pragma unroll
      for (int w = 1; w < kWarps; ++w) {
        if (red[w] > bv || (red[w] == bv && red_id[w] < bi)) {
          bv = red[w]; bi = red_id[w];
        }
      }
      red[0] = bv; red_id[0] = bi;
    }
    __syncthreads();
    const int winner = red_id[0];
    if (tid == winner) {
      taken[tid] = 1;
      // Staging: [row, rank] = id, [row, rank+8] = probability.
      row_top8[row * 16 + rank] = static_cast<float>(tid);
      row_top8[row * 16 + 8 + rank] = expf(row_f32[tid] - row_max) / denom;
    }
    __syncthreads();
  }
}

// Stage B: one thread per expert accumulates its probability mass across
// rows in fixed row order (deterministic).
template <Device::Type kDevice>
__global__ void CloverCoresetMassKernel(const float* __restrict__ row_top8,
                                        float* __restrict__ mass,
                                        int64_t tokens, int64_t blocks,
                                        int width) {
  const int expert = static_cast<int>(threadIdx.x);
  const int64_t block = static_cast<int64_t>(blockIdx.x);
  if (block >= blocks || expert >= kCloverCoresetExperts) return;
  float total = 0.0f;
  for (int t = 0; t < width; ++t) {
    const float* slot = row_top8 + (block * width + t) * 16;
#pragma unroll
    for (int rank = 0; rank < 8; ++rank) {
      if (static_cast<int>(slot[rank]) == expert) total += slot[8 + rank];
    }
  }
  mass[block * kCloverCoresetExperts + expert] = total;
}

// One block (256 threads) per routing block. Deterministic bitonic sort
// of (mass, expert_id) pairs descending by mass (ties prefer smaller id,
// matching the Triton bitcast key), a prefix sum of the normalized
// probabilities, and the Triton keep rule: keep element i while the
// cumulative probability BEFORE i stays below top_p (top_p >= 1 keeps
// every positive-mass expert; the highest-mass expert is always kept,
// which is what the reference's sorted-position-0 OR amounts to).
template <Device::Type kDevice>
__global__ void CloverCoresetAllowKernel(const float* __restrict__ mass,
                                         int8_t* __restrict__ allow,
                                         int64_t blocks, float top_p) {
  constexpr int kExperts = kCloverCoresetExperts;  // 256 (power of two)
  const int64_t block = static_cast<int64_t>(blockIdx.x);
  if (block >= blocks) return;
  const int tid = static_cast<int>(threadIdx.x);
  if (tid >= kExperts) return;

  extern __shared__ float smem[];
  float* val = smem;                 // [256] mass being sorted
  int* idx = reinterpret_cast<int*>(smem + kExperts);
  __shared__ float prefix[kExperts];  // static shared is fine

  val[tid] = mass[block * kExperts + tid];
  idx[tid] = tid;
  __syncthreads();

  // Bitonic sort descending by (val, -idx). 256 elements, 256 threads:
  // each thread owns one slot; every compare-exchange step is a block
  // barrier sync with pairs (i, j) computed from the step parameters.
  for (int k = 2; k <= kExperts; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      const int partner = tid ^ j;
      if (partner > tid) {  // only one side of the pair acts
        const bool ascending_block =
            ((tid & k) == 0);  // block direction for this element
        const float a = val[tid], b = val[partner];
        const int ia = idx[tid], ib = idx[partner];
        // sorted descending overall: swap if (a,b) is out of order for
        // the block direction. For descending output with bitonic sort,
        // direction flips per (tid & k); using !ascending for desc.
        const bool out_of_order =
            ascending_block ? (a < b || (a == b && ia > ib))
                            : (a > b || (a == b && ia < ib));
        if (out_of_order) {
          val[tid] = b; val[partner] = a;
          idx[tid] = ib; idx[partner] = ia;
        }
      }
      __syncthreads();
    }
  }

  // Prefix sums of normalized probabilities in sorted order.
  float total_acc = 0.0f;
  for (int e = 0; e < kExperts; ++e) total_acc += val[e];
  const float total = total_acc > 1.0e-9f ? total_acc : 1.0e-9f;

  if (tid == 0) {
    float cum = 0.0f;
    for (int i = 0; i < kExperts; ++i) {
      prefix[i] = cum;  // probability mass strictly before element i
      cum += val[i] / total;
    }
  }
  __syncthreads();

  const bool keep_positive_only = top_p >= 1.0f;
  const bool keep = keep_positive_only
                        ? val[tid] > 0.0f
                        : prefix[tid] < top_p;
  allow[block * kExperts + idx[tid]] = static_cast<int8_t>(keep ? 1 : 0);
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
