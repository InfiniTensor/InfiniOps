#ifndef INFINI_OPS_CUDA_CLOVER_RECURRENT_GDN_KERNEL_CUH_
#define INFINI_OPS_CUDA_CLOVER_RECURRENT_GDN_KERNEL_CUH_

#include <cstdint>

#include "native/cuda/caster.cuh"

namespace infini::ops {

// Shared-layer SIMT kernel for CloverRecurrentGdn, mirroring the Triton
// reference: per (request, head, value tile), sequentially update the GDN
// state [V, K] from the initial slot and write the final state back:
//   q,k L2-normalized (eps 1e-6), q scaled by 0.08838834764831845 (=1/sqrt(128))
//   state *= exp(g); delta = (v - state·k) * beta; state += delta⊗k; y = state·q
// Deterministic: serial token loop, one thread per value row.
// Grid: x = requests, y = heads, z = value tiles (128 / BV).
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
    int value_dim, int bv) {
  const int request = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  const int value_tile = static_cast<int>(blockIdx.z);

  const int begin = offsets[request];
  const int end = offsets[request + 1];
  const int init = initial[request];
  const int fin = final_idx[request];
  if (!(begin >= 0 && end >= begin && end <= tokens && init >= 0 &&
        init < pool && fin >= 0 && fin < pool))
    return;

  const int value_row = value_tile * bv + static_cast<int>(threadIdx.x);
  if (value_row >= value_dim) return;

  const int q_head = head / (gridDim.y / num_k_heads);  // v-heads share q/k heads
  const int k_head = q_head;

  float* state_row =
      state + static_cast<int64_t>(init) * ss + head * sh +
      static_cast<int64_t>(value_row) * sv;

  for (int token = begin; token < end; ++token) {
    // k row L2 norm (fp32, over 128 keys) — every thread recomputes it; kept
    // simple and deterministic.
    float k_vec[128];
    float k_sq = 0.0f;
    const Data* k_row = k + static_cast<int64_t>(token) * ks + k_head * kh;
    for (int kk = 0; kk < 128; ++kk) {
      k_vec[kk] = Caster<kDevice>::template Cast<float>(k_row[kk]);
      k_sq += k_vec[kk] * k_vec[kk];
    }
    const float k_norm = rsqrtf(k_sq + 1.0e-6f);
    for (int kk = 0; kk < 128; ++kk) k_vec[kk] *= k_norm;

    const float decay = expf(g[token * gs + head * gh]);
    const float b = beta[token * bs + head * bh];
    const float v_row = Caster<kDevice>::template Cast<float>(
        v[static_cast<int64_t>(token) * vs + head * vh + value_row]);

    float remembered = 0.0f;
    for (int kk = 0; kk < 128; ++kk) {
      state_row[kk * sk] *= decay;
      remembered += state_row[kk * sk] * k_vec[kk];
    }
    const float delta = (v_row - remembered) * b;
    float y = 0.0f;
    const Data* q_row =
        q + static_cast<int64_t>(token) * qs + q_head * qh;
    float q_sq = 0.0f;
    float q_vec[128];
    for (int kk = 0; kk < 128; ++kk) {
      q_vec[kk] = Caster<kDevice>::template Cast<float>(q_row[kk]);
      q_sq += q_vec[kk] * q_vec[kk];
    }
    const float q_norm = rsqrtf(q_sq + 1.0e-6f) * 0.08838834764831845f;
    for (int kk = 0; kk < 128; ++kk) {
      q_vec[kk] *= q_norm;
      state_row[kk * sk] += delta * k_vec[kk];
      y += state_row[kk * sk] * q_vec[kk];
    }
    out[static_cast<int64_t>(token) * os + head * oh + value_row] =
        Caster<kDevice>::template Cast<Data>(y);
  }

  // Write the final state back to the final slot (may equal the initial one).
  if (fin != init) {
    float* dst_row = state + static_cast<int64_t>(fin) * ss + head * sh +
                     static_cast<int64_t>(value_row) * sv;
    for (int kk = 0; kk < 128; ++kk)
      dst_row[kk * sk] = state_row[kk * sk];
  }
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_RECURRENT_GDN_KERNEL_CUH_
