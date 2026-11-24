#ifndef INFINI_OPS_CUDA_CLOVER_RECURRENT_GDN_KERNEL_CUH_
#define INFINI_OPS_CUDA_CLOVER_RECURRENT_GDN_KERNEL_CUH_

#include <cstdint>

#include "native/cuda/caster.cuh"

namespace infini::ops {

// Shared-layer SIMT kernel for CloverRecurrentGdn, mirroring the Triton
// reference. Warp-cooperative layout: one warp owns one value row of the
// [V=128, K=128] state; the 32 lanes split the K axis, so every dot
// product over K becomes a fixed-order warp shuffle reduction
// (deterministic) and the normalized k vector is staged once per token in
// shared memory instead of being recomputed per row.
//   state *= exp(g); delta = (v - state·k) * beta; state += delta⊗k; y = state·q
// Grid: x = requests, y = value heads; block = 4 warps (4 value rows each).
template <Device::Type kDevice, typename Data>
__global__ void CloverRecurrentGdnKernel(
    const Data* __restrict__ q, const Data* __restrict__ k,
    const Data* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ initial, const int32_t* __restrict__ final_idx,
    float* __restrict__ state, Data* __restrict__ out, int64_t qs, int64_t qh,
    int64_t ks, int64_t kh, int64_t vs, int64_t vh, int64_t gs, int64_t gh,
    int64_t bs, int64_t bh, int64_t ss, int64_t sh, int64_t sv, int64_t sk,
    int64_t os, int64_t oh, int64_t tokens, int64_t pool, int num_k_heads,
    int value_dim) {
  constexpr int kDim = 128;                 // K == V == 128 (validated)
  constexpr int kLanes = 32;                // warp width on supported backends
  constexpr int kPerLane = kDim / kLanes;   // 4 keys per lane
  constexpr int kWarps = 4;                 // value rows per block per step

  const int request = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  const int warp = static_cast<int>(threadIdx.x) / kLanes;
  const int lane = static_cast<int>(threadIdx.x) % kLanes;
  if (warp >= kWarps) return;
  const int value_row =
      static_cast<int>(blockIdx.z) * kWarps + warp;  // absolute row 0..127

  const int begin = offsets[request];
  const int end = offsets[request + 1];
  const int init = initial[request];
  const int fin = final_idx[request];
  if (!(begin >= 0 && end >= begin && end <= tokens && init >= 0 &&
        init < pool && fin >= 0 && fin < pool))
    return;

  const int k_head =
      head / max(1, (int)gridDim.y / max(1, num_k_heads));

  extern __shared__ float smem[];
  float* k_norm = smem;                  // [kWarps][kDim]

  float* state_row =
      state + static_cast<int64_t>(init) * ss + head * sh +
      static_cast<int64_t>(value_row) * sv;
  float* final_row =
      state + static_cast<int64_t>(fin) * ss + head * sh +
      static_cast<int64_t>(value_row) * sv;

  float st[kPerLane];
#pragma unroll
  for (int e = 0; e < kPerLane; ++e)
    st[e] = state_row[(lane * kPerLane + e) * sk];

  for (int token = begin; token < end; ++token) {
    // ---- k: cooperative load + fixed-order square-sum reduction ----
    const Data* k_row =
        k + static_cast<int64_t>(token) * ks + k_head * kh;
    float kv[kPerLane];
    float k_sq = 0.0f;
#pragma unroll
    for (int e = 0; e < kPerLane; ++e) {
      kv[e] = Caster<kDevice>::template Cast<float>(
          k_row[lane * kPerLane + e]);
      k_sq += kv[e] * kv[e];
    }
#pragma unroll
    for (int off = kLanes / 2; off > 0; off >>= 1)
      k_sq += __shfl_xor_sync(0xffffffffu, k_sq, off);
    const float k_scale = rsqrtf(k_sq + 1.0e-6f);
#pragma unroll
    for (int e = 0; e < kPerLane; ++e) {
      kv[e] *= k_scale;
      k_norm[warp * kDim + lane * kPerLane + e] = kv[e];
    }
    __syncwarp();

    // ---- decay + remembered = state · k ----
    const float decay = expf(g[token * gs + head * gh]);
    const float b = beta[token * bs + head * bh];
    float part = 0.0f;
#pragma unroll
    for (int e = 0; e < kPerLane; ++e) {
      st[e] *= decay;
      part += st[e] * kv[e];
    }
#pragma unroll
    for (int off = kLanes / 2; off > 0; off >>= 1)
      part += __shfl_xor_sync(0xffffffffu, part, off);
    const float remembered = part;

    // ---- delta rule update ----
    const float v_row = Caster<kDevice>::template Cast<float>(
        v[static_cast<int64_t>(token) * vs + head * vh + value_row]);
    const float delta = (v_row - remembered) * b;

    const Data* q_row =
        q + static_cast<int64_t>(token) * qs + k_head * qh;
    float qv[kPerLane];
    float q_sq = 0.0f;
#pragma unroll
    for (int e = 0; e < kPerLane; ++e) {
      qv[e] = Caster<kDevice>::template Cast<float>(
          q_row[lane * kPerLane + e]);
      q_sq += qv[e] * qv[e];
    }
#pragma unroll
    for (int off = kLanes / 2; off > 0; off >>= 1)
      q_sq += __shfl_xor_sync(0xffffffffu, q_sq, off);
    const float q_scale = rsqrtf(q_sq + 1.0e-6f) * 0.08838834764831845f;

    // ---- state += delta * k; y = state · q (same fixed order) ----
    float y = 0.0f;
#pragma unroll
    for (int e = 0; e < kPerLane; ++e) {
      st[e] += delta * k_norm[warp * kDim + lane * kPerLane + e];
      qv[e] *= q_scale;
      y += st[e] * qv[e];
    }
#pragma unroll
    for (int off = kLanes / 2; off > 0; off >>= 1)
      y += __shfl_xor_sync(0xffffffffu, y, off);

    if (lane == 0) {
      out[static_cast<int64_t>(token) * os + head * oh + value_row] =
          Caster<kDevice>::template Cast<Data>(y);
    }
  }

  // Final-state writeback (may equal the initial slot).
#pragma unroll
  for (int e = 0; e < kPerLane; ++e)
    final_row[(lane * kPerLane + e) * sk] = st[e];
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_RECURRENT_GDN_KERNEL_CUH_
