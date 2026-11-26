#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace infini::ops {
namespace v4_route {

// Reproduce the Triton gate's stable log1p expansion and HCU math.
static __device__ float probability(float x) {
  const float exponent =
      __builtin_amdgcn_exp2f(-fabsf(x) * 1.4426950408889634f);
  volatile float shifted = 1.0f + exponent;
  const float log2 = __builtin_amdgcn_logf(shifted);
  const float ln2_hi = __uint_as_float(0x3f317217);
  const float ln2_lo = __uint_as_float(0x3377d1cf);
  const float product = __fmul_rn(log2, ln2_hi);
  float correction = fmaf(log2, ln2_hi, -product);
  correction = fmaf(log2, ln2_lo, correction);
  volatile float natural_log = fmaf(log2, ln2_hi, correction);
  volatile float numerator = exponent * natural_log;
  const float log1p = shifted == 1.0f ? exponent : numerator / (shifted - 1.0f);
  return __builtin_amdgcn_sqrtf(fmaxf(x, 0.0f) + log1p);
}

// HCU's hash gate uses stable sqrt(softplus), then a six-lane warp sum.
static __device__ float hash_probability(float x) {
  return sqrtf(fmaxf(x, 0.f) + log1pf(expf(-fabsf(x))));
}

template <class Token>
__device__ void RouteRow(const float* logits, const float* bias,
                         const Token* tokens, const int64_t* hash, float scale,
                         float* weights, int64_t* ids, size_t rows,
                         size_t experts, size_t k, size_t vocab, size_t row) {
  const auto* x = logits + row * experts;
  auto* selected = ids + row * k;
  auto* w = weights + row * k;
  const auto token = tokens ? static_cast<int64_t>(tokens[row]) : -1;
  float norm = 0;
  for (size_t i = 0; i < k; ++i) {
    int64_t id = -1;
    if (hash) {
      if (token >= 0 && size_t(token) < vocab) id = hash[size_t(token) * k + i];
    } else {
      float best = -INFINITY;
      for (size_t j = 0; j < experts; ++j) {
        const float rank = probability(x[j]) + (bias ? bias[j] : 0);
        if (rank <= best) continue;
        bool used = false;
        for (size_t prev = 0; prev < i; ++prev)
          used |= selected[prev] == int64_t(j);
        if (!used) {
          best = rank;
          id = j;
        }
      }
    }
    const bool valid = id >= 0 && size_t(id) < experts;
    selected[i] = valid ? id : -1;
    w[i] = valid ? (hash ? hash_probability(x[id]) : probability(x[id])) : 0.f;
    norm += w[i];
  }
  if (k == 6) {
    // The two reference kernels pack the six weights into lanes differently.
    if (hash) {
      norm = ((w[0] + w[4]) + w[2]) + ((w[1] + w[5]) + w[3]);
    } else {
      volatile float first = (((w[0] + w[1]) + w[2]) + w[3]);
      volatile float second = w[4] + w[5];
      norm = first + second;
    }
  }
  for (size_t i = 0; i < k; ++i) w[i] = norm > 0 ? (w[i] / norm) * scale : 0.f;
}
template <class Token>
__global__ void route(const float* logits, const float* bias,
                      const Token* tokens, const int64_t* hash, float scale,
                      float* weights, int64_t* ids, size_t rows, size_t experts,
                      size_t k, size_t vocab) {
  for (size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += gridDim.x * blockDim.x) {
    RouteRow(logits, bias, tokens, hash, scale, weights, ids, rows, experts, k,
             vocab, row);
  }
}

__device__ __forceinline__ uint64_t RouteKey(float score, unsigned expert) {
  // Comparisons treat both signed zeros as equal; preserve expert-id tie order.
  const uint32_t bits = score == 0.0f ? 0u : __float_as_uint(score);
  const uint32_t ordered = bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
  return (static_cast<uint64_t>(ordered) << 32) |
         static_cast<uint32_t>(~expert);
}

__device__ __forceinline__ void SortPair(uint64_t& a, uint64_t& b) {
  const uint64_t hi = a > b ? a : b;
  b = a > b ? b : a;
  a = hi;
}

static __global__ void RouteWave64(const float* logits, const float* bias,
                                   float scale, float* weights, int64_t* ids,
                                   size_t rows) {
  constexpr int kWave = 64;
  constexpr unsigned long long kMask = 0xffffffffffffffffull;
  const int lane = threadIdx.x % kWave;
  const size_t waves = blockDim.x / kWave;
  for (size_t row = blockIdx.x * waves + threadIdx.x / kWave; row < rows;
       row += gridDim.x * waves) {
    uint64_t keys[4];
    bool unusual = false;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const unsigned expert = lane + i * kWave;
      const float score = probability(logits[row * 256 + expert]) +
                          (bias ? bias[expert] : 0.0f);
      unusual |= !isfinite(score);
      keys[i] = RouteKey(score, expert);
    }
    if (__any_sync(kMask, unusual)) {
      if (lane == 0) {
        RouteRow<int64_t>(logits, bias, nullptr, nullptr, scale, weights, ids,
                          rows, 256, 6, 0, row);
      }
      continue;
    }
    SortPair(keys[0], keys[1]);
    SortPair(keys[2], keys[3]);
    SortPair(keys[0], keys[2]);
    SortPair(keys[1], keys[3]);
    SortPair(keys[1], keys[2]);
    int selected = -1;
#pragma unroll
    for (int k = 0; k < 6; ++k) {
      uint64_t best = keys[0];
#pragma unroll
      for (int offset = kWave / 2; offset; offset >>= 1) {
        const uint64_t other = __shfl_xor_sync(kMask, best, offset, kWave);
        best = best > other ? best : other;
      }
      const unsigned expert = ~static_cast<uint32_t>(best);
      if (lane == k) selected = static_cast<int>(expert);
      if (keys[0] == best) {
        keys[0] = keys[1];
        keys[1] = keys[2];
        keys[2] = keys[3];
        keys[3] = 0;
      }
    }
    const float value =
        lane < 6 ? probability(logits[row * 256 + selected]) : 0.0f;
    const float w0 = __shfl_sync(kMask, value, 0, kWave);
    const float w1 = __shfl_sync(kMask, value, 1, kWave);
    const float w2 = __shfl_sync(kMask, value, 2, kWave);
    const float w3 = __shfl_sync(kMask, value, 3, kWave);
    const float w4 = __shfl_sync(kMask, value, 4, kWave);
    const float w5 = __shfl_sync(kMask, value, 5, kWave);
    volatile float first = (((w0 + w1) + w2) + w3);
    volatile float second = w4 + w5;
    const float norm = first + second;
    if (lane < 6) {
      weights[row * 6 + lane] = norm > 0 ? (value / norm) * scale : 0.0f;
      ids[row * 6 + lane] = selected;
    }
  }
}

}  // namespace v4_route

}  // namespace infini::ops
