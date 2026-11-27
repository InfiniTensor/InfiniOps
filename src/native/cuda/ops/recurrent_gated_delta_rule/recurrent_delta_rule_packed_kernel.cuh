#ifndef INFINI_OPS_CUDA_RECURRENT_DELTA_RULE_PACKED_KERNEL_CUH_
#define INFINI_OPS_CUDA_RECURRENT_DELTA_RULE_PACKED_KERNEL_CUH_

#include <cstdint>

#include "data_type.h"
#include "native/cuda/caster.cuh"

namespace infini::ops {
namespace recurrent_gated_delta_rule_detail {

__device__ inline int64_t PackedIndex(const void* p, bool i64, int i,
                                      int64_t fallback) {
  return p ? (i64 ? static_cast<const int64_t*>(p)[i]
                  : static_cast<int64_t>(static_cast<const int32_t*>(p)[i]))
           : fallback;
}

// Sequential gated delta rule over packed request sequences with an FP32
// state pool: the math of the single-token kernel, run token by token.
// Warp-cooperative layout: one warp owns one value row of the [V=128, K=128]
// state; the 32 lanes split the K axis, so every dot product over K becomes a
// fixed-order warp shuffle reduction (deterministic) and the normalized k
// vector is staged once per token in shared memory instead of being recomputed
// per row. The state of the row lives in four registers per lane, read from
// the initial slot and written once to the final slot, which leaves a
// differing initial slot unchanged.
//   state *= exp(g); delta = (v - state·k) * beta; state += delta⊗k; y =
//   state·q
// Grid: x = requests, y = value heads, z = value-row tiles; block = kWarps
// warps of 32 lanes. `kDim` is the head dimension; the query scale below is
// written for 128.
template <Device::Type kDevice, typename Data, int kDim, int kWarps>
__global__ void PackedRecurrentGatedDeltaRuleKernel(
    const Data* __restrict__ q, const Data* __restrict__ k,
    const Data* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, const void* __restrict__ cu_seqlens,
    const void* __restrict__ initial_indices,
    const void* __restrict__ final_indices, bool offsets_i64, bool initial_i64,
    bool final_i64, bool norm, float* __restrict__ state,
    Data* __restrict__ out, int64_t qs, int64_t qh, int64_t ks, int64_t kh,
    int64_t vs, int64_t vh, int64_t gs, int64_t gh, int64_t bs, int64_t bh,
    int64_t ss, int64_t sh, int64_t sv, int64_t sk, int64_t os, int64_t oh,
    int64_t tokens, int64_t pool, int num_k_heads) {
  constexpr int kLanes = 32;               // warp width on supported backends
  constexpr int kPerLane = kDim / kLanes;  // keys per lane
  static_assert(kDim == 128, "the query scale is written for D == 128");
  constexpr float kQueryScale = 0.08838834764831845f;

  const int request = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  const int warp = static_cast<int>(threadIdx.x) / kLanes;
  const int lane = static_cast<int>(threadIdx.x) % kLanes;
  if (warp >= kWarps) return;
  const int value_row =
      static_cast<int>(blockIdx.z) * kWarps + warp;  // absolute row 0..V-1

  const int64_t begin = PackedIndex(cu_seqlens, offsets_i64, request, 0);
  const int64_t end = PackedIndex(cu_seqlens, offsets_i64, request + 1, 0);
  const int64_t initial =
      PackedIndex(initial_indices, initial_i64, request, -1);
  const int64_t final_slot = PackedIndex(final_indices, final_i64, request, -1);
  const bool range_ok = begin >= 0 && end >= begin && end <= tokens;
  const bool slots_ok =
      initial >= 0 && initial < pool && final_slot >= 0 && final_slot < pool;
  if (!range_ok) return;
  if (!slots_ok) {
    // Same as the batched form: an unusable slot zeroes the output only.
    const Data zero = Caster<kDevice>::template Cast<Data>(0.0f);
    for (int64_t token = begin; token < end; ++token)
      if (lane == 0) out[token * os + head * oh + value_row] = zero;
    return;
  }

  const int k_head = head / max(1, (int)gridDim.y / max(1, num_k_heads));

  extern __shared__ float smem[];
  float* k_norm = smem;  // [kWarps][kDim]

  float* state_row =
      state + initial * ss + head * sh + static_cast<int64_t>(value_row) * sv;
  float* final_row = state + final_slot * ss + head * sh +
                     static_cast<int64_t>(value_row) * sv;

  float st[kPerLane];
#pragma unroll
  for (int e = 0; e < kPerLane; ++e)
    st[e] = state_row[(lane * kPerLane + e) * sk];

  for (int64_t token = begin; token < end; ++token) {
    // ---- k: cooperative load + fixed-order square-sum reduction ----
    const Data* k_row = k + token * ks + k_head * kh;
    float kv[kPerLane];
    float k_sq = 0.0f;
#pragma unroll
    for (int e = 0; e < kPerLane; ++e) {
      kv[e] = Caster<kDevice>::template Cast<float>(k_row[lane * kPerLane + e]);
      k_sq += kv[e] * kv[e];
    }
#pragma unroll
    for (int off = kLanes / 2; off > 0; off >>= 1)
      k_sq += __shfl_xor_sync(0xffffffffu, k_sq, off);
    const float k_scale = norm ? rsqrtf(k_sq + 1.0e-6f) : 1.0f;
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
        v[token * vs + head * vh + value_row]);
    const float delta = (v_row - remembered) * b;

    const Data* q_row = q + token * qs + k_head * qh;
    float qv[kPerLane];
    float q_sq = 0.0f;
#pragma unroll
    for (int e = 0; e < kPerLane; ++e) {
      qv[e] = Caster<kDevice>::template Cast<float>(q_row[lane * kPerLane + e]);
      q_sq += qv[e] * qv[e];
    }
#pragma unroll
    for (int off = kLanes / 2; off > 0; off >>= 1)
      q_sq += __shfl_xor_sync(0xffffffffu, q_sq, off);
    const float q_scale = (norm ? rsqrtf(q_sq + 1.0e-6f) : 1.0f) * kQueryScale;

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
      out[token * os + head * oh + value_row] =
          Caster<kDevice>::template Cast<Data>(y);
    }
  }

  // Write the final state to the final slot (may equal the initial one).
#pragma unroll
  for (int e = 0; e < kPerLane; ++e)
    final_row[(lane * kPerLane + e) * sk] = st[e];
}

}  // namespace recurrent_gated_delta_rule_detail
}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_RECURRENT_DELTA_RULE_PACKED_KERNEL_CUH_
