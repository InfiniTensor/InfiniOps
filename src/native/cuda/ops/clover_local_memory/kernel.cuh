#ifndef INFINI_OPS_CUDA_CLOVER_LOCAL_MEMORY_KERNEL_CUH_
#define INFINI_OPS_CUDA_CLOVER_LOCAL_MEMORY_KERNEL_CUH_

#include <cstdint>

#include "native/cuda/caster.cuh"

namespace infini::ops {

// Shared-layer SIMT kernel for CloverLocalMemory, mirroring the Triton
// reference (read-only BD-GDN preview). Per (request, value head):
//   1. normalize q rows and block-local k columns (L2, eps 1e-6, fp32)
//   2. scores = (q . k) * logit_scale; masked by local validity (-inf)
//   3. history = q . clean_state[v0:v0+VEC] (state layout [slot,H,V,K], fp32)
//   4. softmax over {local scores} + one history bucket whose logit is
//      history_bias + history_c * log(valid_block_size / reference)
//   5. out = hist * scale * exp(g) * history_weight + softmax . values
// Deterministic: one thread per (query row, value slice), serial loops.
// Grid: x = value tiles (VALUE_DIM/VEC), y = requests * query tiles, z = value heads.
template <Device::Type kDevice, typename Data, typename Index>
__global__ void CloverLocalMemoryKernel(
    const Data* __restrict__ q, const Data* __restrict__ k,
    const Data* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ state, const Index* __restrict__ indices,
    const int8_t* __restrict__ valid, const Data* __restrict__ logit_scale,
    const Data* __restrict__ history_bias_in,
    const Data* __restrict__ history_c_in, Data* __restrict__ out,
    int64_t q_stride_t, int64_t q_stride_h, int64_t q_stride_k,
    int64_t k_stride_t, int64_t k_stride_h, int64_t k_stride_k,
    int64_t v_stride_t, int64_t v_stride_h, int64_t v_stride_v,
    int64_t g_stride_t, int64_t g_stride_h, int64_t state_slot_stride,
    int64_t state_h_stride, int64_t state_v_stride, int64_t state_k_stride,
    int64_t out_t_stride, int64_t out_h_stride, int64_t out_v_stride,
    float scale, float reference_block_size, int block_size, int num_slots,
    int num_q_heads, int num_k_heads, int num_v_heads) {
  constexpr int kDim = 128;   // KEY_DIM == VALUE_DIM == 128 (validated)
  constexpr int kVec = 8;     // value dims per thread

  const int value_tile = static_cast<int>(blockIdx.x);
  const int query_program = static_cast<int>(blockIdx.y);
  const int num_query_tiles = gridDim.y / gridDim.z >= 1
                                  ? (gridDim.y + 0) / 1 / 1
                                  : 1;  // placeholder; host passes 1 tile
  (void)num_query_tiles;
  const int request = query_program;  // host: y = requests * 1 tile
  const int value_head = static_cast<int>(blockIdx.z);

  const int t = static_cast<int>(threadIdx.x);  // query row in block
  if (t >= block_size) return;
  const int v0 = value_tile * kVec;

  const auto cache_index = indices[request];
  const bool request_active = cache_index >= 0 && cache_index < num_slots;
  const auto safe_slot = cache_index >= 0 ? cache_index : 0;
  const int token = request * block_size + t;
  const bool row_active = request_active && (valid[token] != 0);

  const int q_head = num_v_heads > num_q_heads
                         ? value_head / (num_v_heads / num_q_heads) : value_head;
  const int k_head = num_v_heads > num_k_heads
                         ? value_head / (num_v_heads / num_k_heads) : value_head;

  const Data* q_row = q + static_cast<int64_t>(token) * q_stride_t +
                      q_head * q_stride_h;
  const Data* k_base = k + static_cast<int64_t>(request) * block_size * k_stride_t +
                       k_head * k_stride_h;
  const Data* v_base = v + static_cast<int64_t>(request) * block_size * v_stride_t +
                       value_head * v_stride_h;
  const float* state_base =
      state + static_cast<int64_t>(safe_slot) * state_slot_stride +
      value_head * state_h_stride;

  // 1. q row L2-normalized into registers (fp32).
  float qv[kDim];
  float q_sq = 0.0f;
#pragma unroll 8
  for (int kk = 0; kk < kDim; ++kk) {
    qv[kk] = Caster<kDevice>::template Cast<float>(q_row[kk * q_stride_k]);
    q_sq += qv[kk] * qv[kk];
  }
  const float q_norm = rsqrtf(q_sq + 1.0e-6f);
#pragma unroll 8
  for (int kk = 0; kk < kDim; ++kk) qv[kk] *= q_norm;

  // 2. k column norms for the whole block (shared across threads).
  extern __shared__ float k_norm_s[];  // [block_size]
  if (t < block_size) {
    float ks = 0.0f;
    const Data* k_col = k_base + t * k_stride_t;
    for (int kk = 0; kk < kDim; ++kk) {
      const float kv = Caster<kDevice>::template Cast<float>(k_col[kk * k_stride_k]);
      ks += kv * kv;
    }
    k_norm_s[t] = rsqrtf(ks + 1.0e-6f);
  }
  __syncthreads();

  // 3. scores for this row against each local key; track validity stats.
  float scores[32];  // block_size <= 32
  float logits[32];
  const float ls = Caster<kDevice>::template Cast<float>(logit_scale[value_head]);
  for (int j = 0; j < block_size; ++j) {
    const Data* k_col = k_base + j * k_stride_t;
    float dot = 0.0f;
    for (int kk = 0; kk < kDim; ++kk) {
      const float kv = Caster<kDevice>::template Cast<float>(k_col[kk * k_stride_k]) *
                       k_norm_s[j];
      dot += qv[kk] * kv;
    }
    scores[j] = dot;
    logits[j] = valid[request * block_size + j] != 0 && request_active
                    ? dot * ls
                    : -INFINITY;
  }

  // 4. history readout for this thread's v0..v0+7 slice.
  float hist[kVec];
#pragma unroll
  for (int d = 0; d < kVec; ++d) hist[d] = 0.0f;
  if (request_active) {
    for (int kk = 0; kk < kDim; ++kk) {
      const float* st = state_base + kk * state_k_stride;
#pragma unroll
      for (int d = 0; d < kVec; ++d)
        hist[d] += qv[kk] * st[(v0 + d) * state_v_stride];
    }
  }

  // 5. combined softmax with the history bucket.
  float hb = Caster<kDevice>::template Cast<float>(history_bias_in[value_head]);
  const float hc = Caster<kDevice>::template Cast<float>(history_c_in[value_head]);
  int valid_count = 0;
  for (int j = 0; j < block_size; ++j)
    if (valid[request * block_size + j] != 0 && request_active) ++valid_count;
  const float vbs = valid_count > 0 ? static_cast<float>(valid_count) : 1.0f;
  hb += hc * __logf(vbs / reference_block_size);

  float row_max = hb;
  for (int j = 0; j < block_size; ++j)
    if (logits[j] > row_max) row_max = logits[j];

  float denom = expf(hb - row_max);
  float exps[32];
  for (int j = 0; j < block_size; ++j) {
    exps[j] = logits[j] == -INFINITY ? 0.0f : expf(logits[j] - row_max);
    denom += exps[j];
  }
  const float inv_denom = 1.0f / denom;
  const float history_weight = expf(hb - row_max) * inv_denom;

  // 6. output = hist * scale * exp(g) * hw + Σ w_j * values_j
  const float gate = scale * expf(g[token * g_stride_t + value_head * g_stride_h]);
  const float hist_gain = gate * history_weight;
  float result[kVec];
#pragma unroll
  for (int d = 0; d < kVec; ++d) result[d] = hist[d] * hist_gain;

  for (int j = 0; j < block_size; ++j) {
    if (exps[j] == 0.0f) continue;
    const float w = exps[j] * inv_denom;
    const Data* v_row =
        v_base + static_cast<int64_t>(j) * v_stride_t + v0 * v_stride_v;
#pragma unroll
    for (int d = 0; d < kVec; ++d)
      result[d] += w * Caster<kDevice>::template Cast<float>(v_row[d * v_stride_v]);
  }

  Data* out_row = out + static_cast<int64_t>(token) * out_t_stride +
                  value_head * out_h_stride + v0 * out_v_stride;
  if (row_active) {
#pragma unroll
    for (int d = 0; d < kVec; ++d)
      out_row[d * out_v_stride] =
          Caster<kDevice>::template Cast<Data>(result[d]);
  } else {
#pragma unroll
    for (int d = 0; d < kVec; ++d)
      out_row[d * out_v_stride] = Data(0);
  }
  (void)scores;
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_LOCAL_MEMORY_KERNEL_CUH_
