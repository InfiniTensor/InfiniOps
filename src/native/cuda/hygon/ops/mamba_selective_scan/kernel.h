// Hygon-specific Mamba selective scan kernels migrated from the
// InfiniCore indicator-5 submission (zxk branch, 2026-10-31):
// src/infiniop/ops/mamba_selective_scan/hygon/mamba_selective_scan_hygon.cu.
// Ported to the InfiniOps operator-specialization architecture.
// Copyright 2025 InfiniTensor; MIT notice: LICENSES/InfiniCore-MIT.txt.
#ifndef INFINI_OPS_HYGON_MAMBA_SELECTIVE_SCAN_KERNEL_H_
#define INFINI_OPS_HYGON_MAMBA_SELECTIVE_SCAN_KERNEL_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "base/mamba_selective_scan.h"
#include "caster.h"
#include "dispatcher.h"
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/mamba_selective_scan/kernel.h"

namespace infini::ops {
namespace hygon_mamba_scan_detail {
namespace {
template <typename T>
__device__ float to_float(T v) {
  return static_cast<float>(v);
}
template <>
__device__ float to_float<half>(half v) {
  return __half2float(v);
}
template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <typename T>
__device__ T from_float(float v) {
  return static_cast<T>(v);
}
template <>
__device__ half from_float<half>(float v) {
  return __float2half_rn(v);
}
template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

__device__ float softplusf_stable(float x) {
  return x > 20.0f ? x : log1pf(expf(x));
}
__device__ float siluf(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplusf_t1(float x) {
  if (x > 20.0f) return x;
  const float e = __expf(x);
  // Preserve small deltas when adding exp(x) to one would round to one.
  return x < -8.0f ? e * (1.0f - 0.5f * e) : __logf(1.0f + e);
}
__device__ __forceinline__ float siluf_t1(float x) {
  return x / (1.0f + __expf(-x));
}

template <typename T>
__device__ __forceinline__ float round_to_dtype(float v) {
  return v;
}
template <>
__device__ __forceinline__ float round_to_dtype<half>(float v) {
  return __half2float(__float2half_rn(v));
}
template <>
__device__ __forceinline__ float round_to_dtype<__nv_bfloat16>(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

template <typename T>
__device__ __forceinline__ float finish_mamba_output(float state_dot, float x,
                                                     float d, float gate) {
  float y = round_to_dtype<T>(state_dot);
  y = round_to_dtype<T>(y + round_to_dtype<T>(x * d));
  y = round_to_dtype<T>(y * round_to_dtype<T>(siluf(gate)));
  return y;
}

template <typename T>
__device__ __forceinline__ float finish_mamba_output_t1(float state_dot,
                                                        float x, float d,
                                                        float gate) {
  float y = round_to_dtype<T>(state_dot);
  y = round_to_dtype<T>(y + round_to_dtype<T>(x * d));
  y = round_to_dtype<T>(y * round_to_dtype<T>(siluf_t1(gate)));
  return y;
}

constexpr int kWavefrontSize = 64;
constexpr int kWavesPerBlock = 4;
constexpr unsigned long long kWavefrontMask = 0xffffffffffffffffull;

struct AffinePair {
  float a;
  float u;
};

__device__ __forceinline__ AffinePair compose_affine(AffinePair prefix,
                                                     AffinePair next) {
  return {next.a * prefix.a, next.a * prefix.u + next.u};
}

__device__ __forceinline__ AffinePair warp_inclusive_affine(AffinePair value) {
  for (int offset = 1; offset < kWavefrontSize; offset <<= 1) {
    const float previous_a = __shfl_up_sync(kWavefrontMask, value.a, offset);
    const float previous_u = __shfl_up_sync(kWavefrontMask, value.u, offset);
    if ((threadIdx.x & (kWavefrontSize - 1)) >= offset) {
      value = compose_affine({previous_a, previous_u}, value);
    }
  }
  return value;
}

template <typename T, int kThreads = kWavefrontSize * kWavesPerBlock,
          bool kCanonical = false>
__global__ __launch_bounds__(kThreads) void mamba_selective_scan_short16_kernel(
    T *out, const T *x, const T *dt, const T *b, const T *c, const T *a_log,
    const T *d, const T *gate, const T *dt_bias, float *state,
    size_t intermediate_arg, size_t state_size_arg) {
  const size_t intermediate = kCanonical ? 4096 : intermediate_arg;
  const size_t state_size = kCanonical ? 16 : state_size_arg;
  constexpr int kTokens = 16;
  constexpr int kSharedStride = kTokens + (kCanonical ? 4 : 0);
  constexpr int kChannelsPerBlock = kThreads / kTokens;
  const size_t linear = blockIdx.x * kChannelsPerBlock + threadIdx.x / kTokens;
  const size_t ch = linear % intermediate;
  const size_t batch_idx = linear / intermediate;
  const int token = threadIdx.x % kTokens;
  const size_t state_base = linear * state_size;

  extern __shared__ unsigned char smem[];
  T *b_shared = reinterpret_cast<T *>(smem);
  T *c_shared = b_shared + kSharedStride * state_size;
  if constexpr (kCanonical) {
    // Coalesce global B/C loads, then pad transposed shared rows.
    for (size_t index = threadIdx.x; index < kTokens * state_size;
         index += kThreads) {
      const size_t n = index % state_size, t = index / state_size;
      const size_t global = batch_idx * kTokens * state_size + index;
      b_shared[n * kSharedStride + t] = b[global];
      c_shared[n * kSharedStride + t] = c[global];
    }
  } else {
    const size_t state_idx = threadIdx.x / kTokens;
    const size_t global_index =
        (batch_idx * kTokens + token) * state_size + state_idx;
    b_shared[state_idx * kSharedStride + token] = b[global_index];
    c_shared[state_idx * kSharedStride + token] = c[global_index];
  }
  __syncthreads();

  const size_t x_idx = (batch_idx * kTokens + token) * intermediate + ch;
  const float x_value = to_float<T>(x[x_idx]);
  const float dt_value = round_to_dtype<T>(
      softplusf_stable(to_float<T>(dt[x_idx]) + to_float<T>(dt_bias[ch])));
  const float d_value = to_float<T>(d[ch]);
  const float gate_value = to_float<T>(gate[x_idx]);
  const float lane_a_coeff =
      -__expf(to_float<T>(a_log[ch * state_size + token]));
  float y = 0.0f;

#pragma unroll 2
  for (int state_idx = 0; state_idx < 16; ++state_idx) {
    const float a_coeff =
        __shfl_sync(kWavefrontMask, lane_a_coeff, state_idx, kTokens);
    const float b_value =
        to_float<T>(b_shared[state_idx * kSharedStride + token]);
    AffinePair prefix = {__expf(a_coeff * dt_value),
                         dt_value * b_value * x_value};
    for (int offset = 1; offset < kTokens; offset <<= 1) {
      const float previous_a =
          __shfl_up_sync(kWavefrontMask, prefix.a, offset, kTokens);
      const float previous_u =
          __shfl_up_sync(kWavefrontMask, prefix.u, offset, kTokens);
      if (token >= offset) {
        prefix = compose_affine({previous_a, previous_u}, prefix);
      }
    }
    float initial_state = token == 0 ? state[state_base + state_idx] : 0.0f;
    initial_state = __shfl_sync(kWavefrontMask, initial_state, 0, kTokens);
    const float current_state = prefix.a * initial_state + prefix.u;
    y += round_to_dtype<T>(current_state) *
         to_float<T>(c_shared[state_idx * kSharedStride + token]);
    if (token == kTokens - 1) {
      state[state_base + state_idx] = current_state;
    }
  }

  y = finish_mamba_output<T>(y, x_value, d_value, gate_value);
  out[x_idx] = from_float<T>(y);
}

template <typename DataType, unsigned int kThreads>
__global__
__launch_bounds__(kThreads) void mamba_selective_scan_sequential_s16_kernel(
    DataType *out, const DataType *x, const DataType *dt, const DataType *b,
    const DataType *c, const DataType *a_log, const DataType *d,
    const DataType *gate, const DataType *dt_bias, float *state, size_t batch,
    size_t seq_len, size_t intermediate) {
  constexpr int kStateSize = 16;
  constexpr int kChannelsPerBlock = kThreads / kStateSize;
  const int state_index = threadIdx.x % kStateSize;
  const size_t linear =
      blockIdx.x * kChannelsPerBlock + threadIdx.x / kStateSize;
  if (linear >= batch * intermediate) {
    return;
  }
  const size_t channel = linear % intermediate;
  const size_t batch_index = linear / intermediate;
  const size_t state_base = linear * kStateSize;
  const float bias_value = to_float<DataType>(dt_bias[channel]);
  const float d_value =
      state_index == 0 ? to_float<DataType>(d[channel]) : 0.0f;
  const size_t prepared_index =
      (batch_index * seq_len + state_index) * intermediate + channel;
  float prepared_x = 0.0f;
  float prepared_delta = 0.0f;
  float prepared_gate = 0.0f;
  if (state_index < seq_len) {
    prepared_x = to_float<DataType>(x[prepared_index]);
    prepared_delta = round_to_dtype<DataType>(
        softplusf_stable(to_float<DataType>(dt[prepared_index]) + bias_value));
    prepared_gate = round_to_dtype<DataType>(
        siluf(to_float<DataType>(gate[prepared_index])));
  }
  float state_value = state[state_base + state_index];
  const float a_value =
      -expf(to_float<DataType>(a_log[channel * kStateSize + state_index]));
  for (size_t token = 0; token < seq_len; ++token) {
    const size_t input_index =
        (batch_index * seq_len + token) * intermediate + channel;
    const size_t token_base = (batch_index * seq_len + token) * kStateSize;
    const float x_value =
        __shfl_sync(kWavefrontMask, prepared_x, token, kStateSize);
    const float delta =
        __shfl_sync(kWavefrontMask, prepared_delta, token, kStateSize);
    const float gate_value =
        __shfl_sync(kWavefrontMask, prepared_gate, token, kStateSize);
    const float discrete_a = expf(a_value * delta);
    const float b_value = to_float<DataType>(b[token_base + state_index]);
    const float c_value = to_float<DataType>(c[token_base + state_index]);
    state_value = discrete_a * state_value + delta * b_value * x_value;
    float state_dot = round_to_dtype<DataType>(state_value) * c_value;
#pragma unroll
    for (int offset = kStateSize / 2; offset > 0; offset >>= 1) {
      state_dot +=
          __shfl_down_sync(kWavefrontMask, state_dot, offset, kStateSize);
    }
    if (state_index == 0) {
      float result = round_to_dtype<DataType>(state_dot);
      result = round_to_dtype<DataType>(
          result + round_to_dtype<DataType>(x_value * d_value));
      result = round_to_dtype<DataType>(result * gate_value);
      out[input_index] = from_float<DataType>(result);
    }
  }
  state[state_base + state_index] = state_value;
}

template <typename DataType, int kTokens>
__global__
__launch_bounds__(256) void mamba_selective_scan_short_s16_4lane_kernel(
    DataType *out, const DataType *x, const DataType *dt, const DataType *b,
    const DataType *c, const DataType *a_log, const DataType *d,
    const DataType *gate, const DataType *dt_bias, float *state, size_t batch,
    size_t intermediate) {
  constexpr int kStateSize = 16;
  constexpr int kLanesPerChannel = 4;
  constexpr int kStatesPerLane = kStateSize / kLanesPerChannel;
  constexpr int kChannelsPerBlock = 256 / kLanesPerChannel;
  const size_t linear =
      blockIdx.x * kChannelsPerBlock + threadIdx.x / kLanesPerChannel;
  if (linear >= batch * intermediate) {
    return;
  }

  const int lane = threadIdx.x % kLanesPerChannel;
  const size_t channel = linear % intermediate;
  const size_t batch_index = linear / intermediate;
  const size_t state_base = linear * kStateSize;
  const float bias_value = to_float<DataType>(dt_bias[channel]);
  const float d_value = lane == 0 ? to_float<DataType>(d[channel]) : 0.0f;
  float state_value[kStatesPerLane];
  float a_value[kStatesPerLane];
#pragma unroll
  for (int index = 0; index < kStatesPerLane; ++index) {
    const int state_index = index * kLanesPerChannel + lane;
    state_value[index] = state[state_base + state_index];
    a_value[index] =
        -expf(to_float<DataType>(a_log[channel * kStateSize + state_index]));
  }

#pragma unroll
  for (int token = 0; token < kTokens; ++token) {
    const size_t input_index =
        (batch_index * kTokens + token) * intermediate + channel;
    const size_t token_base = (batch_index * kTokens + token) * kStateSize;
    float prepared_x = 0.0f;
    float prepared_delta = 0.0f;
    float prepared_gate = 0.0f;
    if (lane == 0) {
      prepared_x = to_float<DataType>(x[input_index]);
      prepared_delta = round_to_dtype<DataType>(
          kTokens == 1
              ? softplusf_t1(to_float<DataType>(dt[input_index]) + bias_value)
              : softplusf_stable(to_float<DataType>(dt[input_index]) +
                                 bias_value));
      prepared_gate = round_to_dtype<DataType>(
          kTokens == 1 ? siluf_t1(to_float<DataType>(gate[input_index]))
                       : siluf(to_float<DataType>(gate[input_index])));
    }
    const float x_value =
        __shfl_sync(kWavefrontMask, prepared_x, 0, kLanesPerChannel);
    const float delta =
        __shfl_sync(kWavefrontMask, prepared_delta, 0, kLanesPerChannel);
    const float gate_value =
        __shfl_sync(kWavefrontMask, prepared_gate, 0, kLanesPerChannel);
    float state_dot = 0.0f;
#pragma unroll
    for (int index = 0; index < kStatesPerLane; ++index) {
      const int state_index = index * kLanesPerChannel + lane;
      const float b_value = to_float<DataType>(b[token_base + state_index]);
      const float c_value = to_float<DataType>(c[token_base + state_index]);
      state_value[index] = expf(a_value[index] * delta) * state_value[index] +
                           delta * b_value * x_value;
      state_dot += round_to_dtype<DataType>(state_value[index]) * c_value;
    }
    state_dot +=
        __shfl_down_sync(kWavefrontMask, state_dot, 2, kLanesPerChannel);
    state_dot +=
        __shfl_down_sync(kWavefrontMask, state_dot, 1, kLanesPerChannel);
    if (lane == 0) {
      float result = round_to_dtype<DataType>(state_dot);
      result = round_to_dtype<DataType>(
          result + round_to_dtype<DataType>(x_value * d_value));
      result = round_to_dtype<DataType>(result * gate_value);
      out[input_index] = from_float<DataType>(result);
    }
  }

#pragma unroll
  for (int index = 0; index < kStatesPerLane; ++index) {
    state[state_base + index * kLanesPerChannel + lane] = state_value[index];
  }
}

template <typename T, int kMaxTokens>
__global__ void mamba_selective_scan_scalar_short_s16_kernel(
    T *out, const T *x, const T *dt, const T *b, const T *c, const T *a_log,
    const T *d, const T *gate, const T *dt_bias, float *state, size_t batch,
    size_t seq_len, size_t intermediate) {
  constexpr int kStateSize = 16;
  const size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = batch * intermediate;
  if (linear >= total) {
    return;
  }

  const size_t channel = linear % intermediate;
  const size_t batch_index = linear / intermediate;
  const size_t state_base = linear * kStateSize;
  const float bias_value = to_float<T>(dt_bias[channel]);
  const float d_value = to_float<T>(d[channel]);

  float state_value[kStateSize];
  float a_value[kStateSize];
#pragma unroll
  for (int state_index = 0; state_index < kStateSize; ++state_index) {
    state_value[state_index] = state[state_base + state_index];
    const float log_value =
        to_float<T>(a_log[channel * kStateSize + state_index]);
    if constexpr (kMaxTokens == 1 && std::is_same_v<T, __nv_bfloat16>) {
      a_value[state_index] = -__expf(log_value);
    } else {
      a_value[state_index] = -expf(log_value);
    }
  }

  for (size_t token = 0; token < seq_len && token < kMaxTokens; ++token) {
    const size_t input_index =
        (batch_index * seq_len + token) * intermediate + channel;
    const float x_value = to_float<T>(x[input_index]);
    const float delta = round_to_dtype<T>(
        kMaxTokens == 1
            ? softplusf_t1(to_float<T>(dt[input_index]) + bias_value)
            : softplusf_stable(to_float<T>(dt[input_index]) + bias_value));
    const float gate_value = to_float<T>(gate[input_index]);
    const size_t token_base = (batch_index * seq_len + token) * kStateSize;
    float result = 0.0f;

#pragma unroll
    for (int state_index = 0; state_index < kStateSize; ++state_index) {
      const float b_value = to_float<T>(b[token_base + state_index]);
      const float c_value = to_float<T>(c[token_base + state_index]);
      const float exponent = a_value[state_index] * delta;
      const float discrete_a =
          kMaxTokens == 1 && std::is_same_v<T, __nv_bfloat16> ? __expf(exponent)
                                                              : expf(exponent);
      state_value[state_index] =
          discrete_a * state_value[state_index] + delta * b_value * x_value;
      result += round_to_dtype<T>(state_value[state_index]) * c_value;
    }

    result =
        kMaxTokens == 1
            ? finish_mamba_output_t1<T>(result, x_value, d_value, gate_value)
            : finish_mamba_output<T>(result, x_value, d_value, gate_value);
    out[input_index] = from_float<T>(result);
  }

#pragma unroll
  for (int state_index = 0; state_index < kStateSize; ++state_index) {
    state[state_base + state_index] = state_value[state_index];
  }
}

template <typename T>
__global__ void mamba_selective_scan_kernel(
    T *out, const T *x, const T *dt, const T *b, const T *c, const T *a_log,
    const T *d, const T *gate, const T *dt_bias, float *state, size_t batch,
    size_t seq_len, size_t intermediate, size_t state_size) {
  size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = batch * intermediate;
  if (linear >= total) {
    return;
  }
  size_t ch = linear % intermediate;
  size_t batch_idx = linear / intermediate;
  size_t state_base = (batch_idx * intermediate + ch) * state_size;
  const float dt_bias_ch = to_float<T>(dt_bias[ch]);
  const float d_ch = to_float<T>(d[ch]);
  const T *a_ch = a_log + ch * state_size;
  for (size_t t = 0; t < seq_len; ++t) {
    size_t x_idx = (batch_idx * seq_len + t) * intermediate + ch;
    float xt = to_float<T>(x[x_idx]);
    float dtv = round_to_dtype<T>(
        softplusf_stable(to_float<T>(dt[x_idx]) + dt_bias_ch));
    float y = 0.0f;
    const size_t token_base = (batch_idx * seq_len + t) * state_size;
    const T *b_t = b + token_base;
    const T *c_t = c + token_base;
#pragma unroll 4
    for (size_t n = 0; n < state_size; ++n) {
      float a = -expf(to_float<T>(a_ch[n]));
      float discrete_a = expf(a * dtv);
      float bn = to_float<T>(b_t[n]);
      float cn = to_float<T>(c_t[n]);
      float s = discrete_a * state[state_base + n] + dtv * bn * xt;
      state[state_base + n] = s;
      y += round_to_dtype<T>(s) * cn;
    }
    y = finish_mamba_output<T>(y, xt, d_ch, to_float<T>(gate[x_idx]));
    out[x_idx] = from_float<T>(y);
  }
}

template <typename T, int kItems, bool kUseShared>
__global__
__launch_bounds__(kWavefrontSize) void mamba_selective_scan_parallel_kernel(
    T *out, const T *x, const T *dt, const T *b, const T *c, const T *a_log,
    const T *d, const T *gate, const T *dt_bias, float *state, size_t batch,
    size_t seq_len, size_t intermediate, size_t state_size) {
  const size_t linear = blockIdx.x;
  const size_t total = batch * intermediate;
  if (linear >= total) {
    return;
  }

  const size_t ch = linear % intermediate;
  const size_t batch_idx = linear / intermediate;
  const size_t state_base = linear * state_size;
  const size_t lane = threadIdx.x & (kWavefrontSize - 1);
  const T *a_ch = a_log + ch * state_size;
  const float d_ch = to_float<T>(d[ch]);
  const float dt_bias_ch = to_float<T>(dt_bias[ch]);

  extern __shared__ unsigned char smem[];
  const size_t tile_tokens = static_cast<size_t>(kWavefrontSize * kItems);
  T *b_shared = reinterpret_cast<T *>(smem);
  T *c_shared = b_shared + tile_tokens * state_size;
  if constexpr (kUseShared) {
    const size_t valid_tokens = seq_len < tile_tokens ? seq_len : tile_tokens;
    const size_t tile_elements = valid_tokens * state_size;
    for (size_t index = lane; index < tile_elements; index += kWavefrontSize) {
      const size_t token = index / state_size;
      const size_t state_idx = index - token * state_size;
      const size_t global_index =
          (batch_idx * seq_len + token) * state_size + state_idx;
      b_shared[state_idx * tile_tokens + token] = b[global_index];
      c_shared[state_idx * tile_tokens + token] = c[global_index];
    }
    __syncthreads();
  }

  float x_vals[kItems];
  float dt_vals[kItems];
  float gate_vals[kItems];
  size_t token_indices[kItems];
#pragma unroll
  for (int i = 0; i < kItems; ++i) {
    const size_t t = lane * kItems + i;
    token_indices[i] = t;
    if (t < seq_len) {
      const size_t x_idx = (batch_idx * seq_len + t) * intermediate + ch;
      x_vals[i] = to_float<T>(x[x_idx]);
      dt_vals[i] = round_to_dtype<T>(
          softplusf_stable(to_float<T>(dt[x_idx]) + dt_bias_ch));
      gate_vals[i] = to_float<T>(gate[x_idx]);
    } else {
      x_vals[i] = 0.0f;
      dt_vals[i] = 0.0f;
      gate_vals[i] = 0.0f;
    }
  }

  float y_vals[kItems] = {};
  for (size_t n = 0; n < state_size; ++n) {
    float a_coeff = lane == 0 ? -expf(to_float<T>(a_ch[n])) : 0.0f;
    a_coeff = __shfl_sync(kWavefrontMask, a_coeff, 0);
    AffinePair local_prefix[kItems];
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      local_prefix[i] = {1.0f, 0.0f};
      const size_t t = token_indices[i];
      if (t < seq_len) {
        float b_value;
        if constexpr (kUseShared) {
          b_value = to_float<T>(b_shared[n * tile_tokens + t]);
        } else {
          b_value = to_float<T>(b[(batch_idx * seq_len + t) * state_size + n]);
        }
        const AffinePair current = {
            expf(a_coeff * dt_vals[i]),
            dt_vals[i] * b_value * x_vals[i],
        };
        local_prefix[i] =
            i == 0 ? current : compose_affine(local_prefix[i - 1], current);
      } else if (i > 0) {
        local_prefix[i] = local_prefix[i - 1];
      }
    }

    const AffinePair lane_prefix =
        warp_inclusive_affine(local_prefix[kItems - 1]);
    const float previous_a = __shfl_up_sync(kWavefrontMask, lane_prefix.a, 1);
    const float previous_u = __shfl_up_sync(kWavefrontMask, lane_prefix.u, 1);
    const AffinePair prefix_before_lane =
        lane == 0 ? AffinePair{1.0f, 0.0f} : AffinePair{previous_a, previous_u};
    float initial_state = lane == 0 ? state[state_base + n] : 0.0f;
    initial_state = __shfl_sync(kWavefrontMask, initial_state, 0);
    float final_state = initial_state;

#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const size_t t = token_indices[i];
      if (t < seq_len) {
        const AffinePair prefix =
            compose_affine(prefix_before_lane, local_prefix[i]);
        const float current_state = prefix.a * initial_state + prefix.u;
        float c_value;
        if constexpr (kUseShared) {
          c_value = to_float<T>(c_shared[n * tile_tokens + t]);
        } else {
          c_value = to_float<T>(c[(batch_idx * seq_len + t) * state_size + n]);
        }
        y_vals[i] += round_to_dtype<T>(current_state) * c_value;
        final_state = current_state;
      }
    }

    if (lane * kItems <= seq_len - 1 && (lane + 1) * kItems > seq_len - 1) {
      state[state_base + n] = final_state;
    }
  }

#pragma unroll
  for (int i = 0; i < kItems; ++i) {
    const size_t t = token_indices[i];
    if (t < seq_len) {
      const size_t x_idx = (batch_idx * seq_len + t) * intermediate + ch;
      float y =
          finish_mamba_output<T>(y_vals[i], x_vals[i], d_ch, gate_vals[i]);
      out[x_idx] = from_float<T>(y);
    }
  }
}

template <typename T, int kItems, int kGroupWaves = kWavesPerBlock,
          bool kCanonical = false>
__global__
__launch_bounds__(kWavefrontSize *kGroupWaves) void mamba_selective_scan_grouped_shared_kernel(
    T *out, const T *x, const T *dt, const T *b, const T *c, const T *a_log,
    const T *d, const T *gate, const T *dt_bias, float *state, size_t batch,
    size_t seq_len, size_t intermediate_arg, size_t state_size_arg) {
  const size_t intermediate = kCanonical ? 4096 : intermediate_arg;
  const size_t state_size = kCanonical ? 16 : state_size_arg;
  const size_t total = batch * intermediate;

  const size_t block_linear = blockIdx.x;
  const size_t base_linear = block_linear * kGroupWaves;
  if (base_linear >= total) {
    return;
  }
  const size_t wave = threadIdx.x / kWavefrontSize;
  const size_t lane = threadIdx.x & (kWavefrontSize - 1);
  const size_t linear = base_linear + wave;
  const size_t batch_idx = base_linear / intermediate;

  extern __shared__ unsigned char smem[];
  const size_t tile_tokens = static_cast<size_t>(kWavefrontSize * kItems);
  const size_t shared_stride = tile_tokens + (kCanonical ? 4 : 0);
  T *b_shared = reinterpret_cast<T *>(smem);
  T *c_shared = b_shared + shared_stride * state_size;
  const size_t valid_tokens = seq_len < tile_tokens ? seq_len : tile_tokens;
  const size_t tile_elements = valid_tokens * state_size;
  for (size_t index = threadIdx.x; index < tile_elements; index += blockDim.x) {
    const size_t token = index / state_size;
    const size_t state_idx = index - token * state_size;
    const size_t global_index =
        (batch_idx * seq_len + token) * state_size + state_idx;
    b_shared[state_idx * shared_stride + token] = b[global_index];
    c_shared[state_idx * shared_stride + token] = c[global_index];
  }
  __syncthreads();
  if (linear >= total) {
    return;
  }
  const size_t ch = linear % intermediate;
  const size_t state_base = linear * state_size;
  const T *a_ch = a_log + ch * state_size;
  const float d_ch = to_float<T>(d[ch]);
  const float dt_bias_ch = to_float<T>(dt_bias[ch]);

  float x_vals[kItems];
  float dt_vals[kItems];
  float gate_vals[kItems];
  size_t token_indices[kItems];
#pragma unroll
  for (int i = 0; i < kItems; ++i) {
    const size_t t = lane * kItems + i;
    token_indices[i] = t;
    if (t < seq_len) {
      const size_t x_idx = (batch_idx * seq_len + t) * intermediate + ch;
      x_vals[i] = to_float<T>(x[x_idx]);
      dt_vals[i] = round_to_dtype<T>(
          softplusf_stable(to_float<T>(dt[x_idx]) + dt_bias_ch));
      gate_vals[i] = to_float<T>(gate[x_idx]);
    } else {
      x_vals[i] = 0.0f;
      dt_vals[i] = 0.0f;
      gate_vals[i] = 0.0f;
    }
  }

  float y_vals[kItems] = {};
  for (size_t n = 0; n < state_size; ++n) {
    float a_coeff = lane == 0 ? -expf(to_float<T>(a_ch[n])) : 0.0f;
    a_coeff = __shfl_sync(kWavefrontMask, a_coeff, 0);
    AffinePair local_prefix[kItems];
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      local_prefix[i] = {1.0f, 0.0f};
      const size_t t = token_indices[i];
      if (t < seq_len) {
        const float b_value = to_float<T>(b_shared[n * shared_stride + t]);
        const AffinePair current = {
            kItems == 2 ? __expf(a_coeff * dt_vals[i])
                        : expf(a_coeff * dt_vals[i]),
            dt_vals[i] * b_value * x_vals[i],
        };
        local_prefix[i] =
            i == 0 ? current : compose_affine(local_prefix[i - 1], current);
      } else if (i > 0) {
        local_prefix[i] = local_prefix[i - 1];
      }
    }

    const AffinePair lane_prefix =
        warp_inclusive_affine(local_prefix[kItems - 1]);
    const float previous_a = __shfl_up_sync(kWavefrontMask, lane_prefix.a, 1);
    const float previous_u = __shfl_up_sync(kWavefrontMask, lane_prefix.u, 1);
    const AffinePair prefix_before_lane =
        lane == 0 ? AffinePair{1.0f, 0.0f} : AffinePair{previous_a, previous_u};
    float initial_state = lane == 0 ? state[state_base + n] : 0.0f;
    initial_state = __shfl_sync(kWavefrontMask, initial_state, 0);
    float final_state = initial_state;

#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const size_t t = token_indices[i];
      if (t < seq_len) {
        const AffinePair prefix =
            compose_affine(prefix_before_lane, local_prefix[i]);
        const float current_state = prefix.a * initial_state + prefix.u;
        const float c_value = to_float<T>(c_shared[n * shared_stride + t]);
        y_vals[i] += round_to_dtype<T>(current_state) * c_value;
        final_state = current_state;
      }
    }

    if (lane * kItems <= seq_len - 1 && (lane + 1) * kItems > seq_len - 1) {
      state[state_base + n] = final_state;
    }
  }

#pragma unroll
  for (int i = 0; i < kItems; ++i) {
    const size_t t = token_indices[i];
    if (t < seq_len) {
      const size_t x_idx = (batch_idx * seq_len + t) * intermediate + ch;
      float y =
          finish_mamba_output<T>(y_vals[i], x_vals[i], d_ch, gate_vals[i]);
      out[x_idx] = from_float<T>(y);
    }
  }
}

__device__ __forceinline__ float decode_expf(float value) {
  return __builtin_amdgcn_exp2f(value * 1.4426950408889634f);
}

__device__ __forceinline__ float decode_logf(float value) {
  return __builtin_amdgcn_logf(value) * 0.6931471805599453f;
}

__device__ __forceinline__ float decode_softplusf(float value) {
  if (value > 20.0f) {
    return value;
  }
  const float exponential = decode_expf(value);
  if (value < -8.0f) {
    return exponential * (1.0f - 0.5f * exponential);
  }
  return decode_logf(1.0f + exponential);
}

template <int kPattern>
__device__ __forceinline__ float decode_quad(float value) {
  return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(value),
                                                 kPattern, 0xf, 0xf, false));
}

template <typename T>
__device__ __forceinline__ T decode_from_float(float value) {
  return from_float<T>(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16
decode_from_float<__nv_bfloat16>(float value) {
  const unsigned int bits = __float_as_uint(value);
  const unsigned int rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
  const unsigned short result =
      (bits & 0x7fffffffu) > 0x7f800000u
          ? 0x7fffu
          : static_cast<unsigned short>(rounded >> 16);
  return __ushort_as_bfloat16(result);
}

template <typename T, int kCount>
struct alignas(sizeof(T) * kCount) DecodeValues {
  T values[kCount];
};

template <typename T>
__device__ __forceinline__ T *decode_offset(T *base, int index) {
  using Byte = std::conditional_t<std::is_const_v<T>, const char, char>;
  const unsigned int offset =
      static_cast<unsigned int>(index) * static_cast<unsigned int>(sizeof(T));
  return reinterpret_cast<T *>(reinterpret_cast<Byte *>(base) + offset);
}

template <typename T, int kThreads, bool kAligned, int kLanes = 4,
          bool kPreserveQuadSum = false, bool kBatchOne = false,
          bool kEarlyState = false>
__global__
__launch_bounds__(kThreads) void mamba_selective_scan_decode_s16_kernel(
    T *out, const T *x, const T *dt, const T *b, const T *c, const T *a_log,
    const T *d, const T *gate, const T *dt_bias, float *state) {
  constexpr int kStatesPerLane = 16 / kLanes;
  using StateVector = DecodeValues<float, kStatesPerLane>;
  using InputVector = DecodeValues<T, kStatesPerLane>;
  const int channel = blockIdx.x * (kThreads / kLanes) + threadIdx.x / kLanes;
  const int lane = threadIdx.x % kLanes;
  const int batch_index = kBatchOne ? 0 : blockIdx.y;
  const int linear = batch_index * 4096 + channel;
  const int state_base = linear * 16 + lane * kStatesPerLane;
  StateVector previous;
  if constexpr (kAligned) {
    previous = *reinterpret_cast<const StateVector *>(
        decode_offset(state, state_base));
  } else {
    __builtin_memcpy(&previous, decode_offset(state, state_base),
                     sizeof(StateVector));
  }
  InputVector log_values;
  InputVector b_values;
  InputVector c_values;
  if constexpr (kAligned) {
    log_values = *reinterpret_cast<const InputVector *>(
        decode_offset(a_log, channel * 16 + lane * kStatesPerLane));
    b_values = *reinterpret_cast<const InputVector *>(
        decode_offset(b, batch_index * 16 + lane * kStatesPerLane));
    c_values = *reinterpret_cast<const InputVector *>(
        decode_offset(c, batch_index * 16 + lane * kStatesPerLane));
  }
  const float x_value = static_cast<float>(*decode_offset(x, linear));
  const float dt_value = static_cast<float>(*decode_offset(dt, linear)) +
                         static_cast<float>(*decode_offset(dt_bias, channel));
  const float delta = decode_softplusf(dt_value);
  const float raw_gate = static_cast<float>(*decode_offset(gate, linear));
  const float gate_value =
      raw_gate * __builtin_amdgcn_rcpf(1.0f + decode_expf(-raw_gate));
  const float scaled_input = delta * x_value;
  const float scaled_delta = delta * 1.4426950408889634f;
  float residual = 0.0f;
  if constexpr (kLanes == 4) {
    residual = static_cast<float>(*decode_offset(d, channel));
    if constexpr (!kEarlyState) asm volatile("" : "+v"(residual));
  }
  StateVector updated;
  float result = 0.0f;
  // Keep the original four-lane (sum0 + sum2) + (sum1 + sum3) reduction order.
  float second_result = 0.0f;
#pragma unroll
  for (int index = 0; index < kStatesPerLane; ++index) {
    const int state_index = lane * kStatesPerLane + index;
    const float log_value =
        kAligned
            ? to_float<T>(log_values.values[index])
            : to_float<T>(*decode_offset(a_log, channel * 16 + state_index));
    const float coefficient = -decode_expf(log_value);
    const float b_value =
        kAligned
            ? to_float<T>(b_values.values[index])
            : to_float<T>(*decode_offset(b, batch_index * 16 + state_index));
    const float c_value =
        kAligned
            ? to_float<T>(c_values.values[index])
            : to_float<T>(*decode_offset(c, batch_index * 16 + state_index));
    const float current = __builtin_amdgcn_exp2f(coefficient * scaled_delta) *
                              previous.values[index] +
                          scaled_input * b_value;
    updated.values[index] = current;
    if constexpr (kPreserveQuadSum) {
      if (index < 4) {
        result += current * c_value;
      } else {
        second_result += current * c_value;
      }
    } else {
      result += current * c_value;
    }
  }
  // B1 can issue state stores before the output reduction to shorten liveness.
  if constexpr (kLanes != 4 || kEarlyState) {
    *reinterpret_cast<StateVector *>(decode_offset(state, state_base)) =
        updated;
  }
  if constexpr (kPreserveQuadSum) {
    result += decode_quad<0xb1>(result);
    second_result += decode_quad<0xb1>(second_result);
    result += second_result;
  } else {
    if constexpr (kLanes >= 4) {
      result += decode_quad<0x4e>(result);
    }
    result += decode_quad<0xb1>(result);
  }
  if (lane == 0) {
    if constexpr (kLanes != 4) {
      residual = static_cast<float>(*decode_offset(d, channel));
    }
    *decode_offset(out, linear) =
        decode_from_float<T>((result + x_value * residual) * gate_value);
  }
  if constexpr (kLanes == 4 && !kEarlyState) {
    if constexpr (kAligned) {
      *reinterpret_cast<StateVector *>(decode_offset(state, state_base)) =
          updated;
    } else {
      __builtin_memcpy(decode_offset(state, state_base), &updated,
                       sizeof(StateVector));
    }
  }
}

template <typename T, int kThreads, bool kAligned, int kLanes = 4,
          bool kPreserveQuadSum = false, bool kBatchOne = false,
          bool kEarlyState = false>
void launch_decode_s16_impl(const MambaScanLayout &p, T *out, const T *x,
                            const T *dt, const T *b, const T *c, const T *a_log,
                            const T *d, const T *gate, const T *dt_bias,
                            float *state, cudaStream_t stream) {
  const dim3 blocks(4096 / (kThreads / kLanes),
                    static_cast<unsigned int>(p.batch));
  mamba_selective_scan_decode_s16_kernel<
      T, kThreads, kAligned, kLanes, kPreserveQuadSum, kBatchOne, kEarlyState>
      <<<blocks, kThreads, 0, stream>>>(out, x, dt, b, c, a_log, d, gate,
                                        dt_bias, state);
}

template <typename T>
void launch_decode_s16(const MambaScanLayout &p, T *out, const T *x,
                       const T *dt, const T *b, const T *c, const T *a_log,
                       const T *d, const T *gate, const T *dt_bias,
                       float *state, cudaStream_t stream) {
  const bool aligned = (reinterpret_cast<uintptr_t>(state) & 15) == 0 &&
                       (reinterpret_cast<uintptr_t>(a_log) & 7) == 0 &&
                       (reinterpret_cast<uintptr_t>(b) & 7) == 0 &&
                       (reinterpret_cast<uintptr_t>(c) & 7) == 0;
  const bool wide_aligned = (reinterpret_cast<uintptr_t>(state) & 31) == 0 &&
                            (reinterpret_cast<uintptr_t>(a_log) & 15) == 0 &&
                            (reinterpret_cast<uintptr_t>(b) & 15) == 0 &&
                            (reinterpret_cast<uintptr_t>(c) & 15) == 0;
  if (p.batch == 1 && aligned) {
    return launch_decode_s16_impl<T, 256, true, 4, false, true, true>(
        p, out, x, dt, b, c, a_log, d, gate, dt_bias, state, stream);
  }
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if (p.batch == 8 && wide_aligned) {
      return launch_decode_s16_impl<T, 512, true, 2, true>(
          p, out, x, dt, b, c, a_log, d, gate, dt_bias, state, stream);
    }
  }
  if (p.batch == 4 && wide_aligned) {
    return launch_decode_s16_impl<T, 512, true, 2>(
        p, out, x, dt, b, c, a_log, d, gate, dt_bias, state, stream);
  }
  if (aligned) {
    return launch_decode_s16_impl<T, 256, true>(p, out, x, dt, b, c, a_log, d,
                                                gate, dt_bias, state, stream);
  }
  if (p.batch == 1) {
    return launch_decode_s16_impl<T, 128, false>(p, out, x, dt, b, c, a_log, d,
                                                 gate, dt_bias, state, stream);
  }
  return launch_decode_s16_impl<T, 256, false>(p, out, x, dt, b, c, a_log, d,
                                               gate, dt_bias, state, stream);
}

// ---------------------------------------------------------------------------
// Host-side launcher: shape-based dispatch migrated from the old Hygon
// descriptor. T=1 decode launches the standalone optimized scan.
// ---------------------------------------------------------------------------
template <typename T, int kItems, int kGroupWaves, bool kCanonical>
void launch_grouped_scan_impl(const MambaScanLayout &p, T *out, const T *x,
                              const T *dt, const T *b, const T *c,
                              const T *a_log, const T *d, const T *gate,
                              const T *dt_bias, float *state,
                              cudaStream_t stream) {
  const size_t states = kCanonical ? 16 : p.states;
  const size_t stride = kWavefrontSize * kItems + (kCanonical ? 4 : 0);
  const size_t shared_bytes = 2 * stride * states * sizeof(T);
  const dim3 blocks((p.batch * p.channels + kGroupWaves - 1) / kGroupWaves);
  mamba_selective_scan_grouped_shared_kernel<T, kItems, kGroupWaves, kCanonical>
      <<<blocks, kWavefrontSize * kGroupWaves, shared_bytes, stream>>>(
          out, x, dt, b, c, a_log, d, gate, dt_bias, state, p.batch, p.length,
          p.channels, p.states);
}

template <typename T, int kItems, int kGroupWaves>
void launch_grouped_scan(const MambaScanLayout &p, T *out, const T *x,
                         const T *dt, const T *b, const T *c, const T *a_log,
                         const T *d, const T *gate, const T *dt_bias,
                         float *state, cudaStream_t stream) {
  if (p.length == 128 && p.channels == 4096 && p.states == 16)
    return launch_grouped_scan_impl<T, kItems, kGroupWaves, true>(
        p, out, x, dt, b, c, a_log, d, gate, dt_bias, state, stream);
  return launch_grouped_scan_impl<T, kItems, kGroupWaves, false>(
      p, out, x, dt, b, c, a_log, d, gate, dt_bias, state, stream);
}

template <typename T, int kThreads, bool kCanonical>
void launch_short16_scan(const MambaScanLayout &p, T *out, const T *x,
                         const T *dt, const T *b, const T *c, const T *a_log,
                         const T *d, const T *gate, const T *dt_bias,
                         float *state, cudaStream_t stream) {
  const size_t stride = 16 + (kCanonical ? 4 : 0);
  const dim3 blocks(p.batch * p.channels / (kThreads / 16));
  mamba_selective_scan_short16_kernel<T, kThreads, kCanonical>
      <<<blocks, kThreads, 2 * stride * p.states * sizeof(T), stream>>>(
          out, x, dt, b, c, a_log, d, gate, dt_bias, state, p.channels,
          p.states);
}

template <typename T>
void launch_scan_dispatch(const MambaScanLayout &p, T *out, const T *x,
                          const T *dt, const T *b, const T *c, const T *a_log,
                          const T *d, const T *gate, const T *dt_bias,
                          float *state, cudaStream_t stream) {
  const size_t batch = p.batch, length = p.length;
  const size_t channels = p.channels, states = p.states;
  const size_t total = batch * channels;

  // F16/BF16 decode: single token, state 16, channels 4096.
  if constexpr (!std::is_same_v<T, float>) {
    if (length == 1 && states == 16 && channels == 4096 &&
        (batch == 1 || batch == 4 || batch == 8)) {
      launch_decode_s16<T>(p, out, x, dt, b, c, a_log, d, gate, dt_bias, state,
                           stream);
      return;
    }
  }

  // F16 (B 1/4/8) / BF16 (B 1) short-s16 4-lane decode.
  if (((std::is_same_v<T, __half> &&
        (batch == 1 || batch == 4 || batch == 8)) ||
       (std::is_same_v<T, __nv_bfloat16> && batch == 1)) &&
      length == 1 && states == 16 && channels % 64 == 0) {
    constexpr int kThreads = 256;
    constexpr int kChannelsPerBlock = kThreads / 4;
    const dim3 blocks((total + kChannelsPerBlock - 1) / kChannelsPerBlock);
    mamba_selective_scan_short_s16_4lane_kernel<T, 1>
        <<<blocks, kThreads, 0, stream>>>(out, x, dt, b, c, a_log, d, gate,
                                          dt_bias, state, batch, channels);
    return;
  }

  // Scalar short-s16 kernel for B=4 short sequences and BF16 B 4/8 T=1.
  if (((batch == 4 && (length == 3 || length == 4 || length == 7)) ||
       (std::is_same_v<T, __nv_bfloat16> && (batch == 4 || batch == 8) &&
        length == 1) ||
       (std::is_same_v<T, __half> && batch == 1 &&
        (length == 3 || length == 4 || length == 7))) &&
      states == 16 && channels % 16 == 0) {
    const size_t threads = total <= 1024 ? 128 : 256;
    const dim3 blocks((total + threads - 1) / threads);
    if (length == 1) {
      mamba_selective_scan_scalar_short_s16_kernel<T, 1>
          <<<blocks, threads, 0, stream>>>(out, x, dt, b, c, a_log, d, gate,
                                           dt_bias, state, batch, length,
                                           channels);
    } else {
      mamba_selective_scan_scalar_short_s16_kernel<T, 7>
          <<<blocks, threads, 0, stream>>>(out, x, dt, b, c, a_log, d, gate,
                                           dt_bias, state, batch, length,
                                           channels);
    }
    return;
  }

  // BF16 sequential s16 for 1 < length < 16.
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if (length > 0 && length < 16 && states == 16 && channels % 16 == 0) {
      constexpr size_t kThreads = 256;
      constexpr size_t kChannelsPerBlock = kThreads / 16;
      const dim3 blocks((total + kChannelsPerBlock - 1) / kChannelsPerBlock);
      mamba_selective_scan_sequential_s16_kernel<T, kThreads>
          <<<blocks, kThreads, 0, stream>>>(out, x, dt, b, c, a_log, d, gate,
                                            dt_bias, state, batch, length,
                                            channels);
      return;
    }
  }

  // Keep dynamic-channel layouts on the original staging path.
  if (length == 16 && states == 16 && channels % 16 == 0) {
    if (channels == 4096) {
      if constexpr (std::is_same_v<T, half>) {
        if (batch == 1)
          return launch_short16_scan<T, 512, true>(
              p, out, x, dt, b, c, a_log, d, gate, dt_bias, state, stream);
      }
      return launch_short16_scan<T, 256, true>(p, out, x, dt, b, c, a_log, d,
                                               gate, dt_bias, state, stream);
    }
    return launch_short16_scan<T, 256, false>(p, out, x, dt, b, c, a_log, d,
                                              gate, dt_bias, state, stream);
  }

  // 16 < length <= 128: parallel / grouped-shared kernels.
  if (length >= 16 && length <= 128) {
    const bool use_grouped_shared =
        channels % kWavesPerBlock == 0 && states <= 64;
    if (length <= 64) {
      const size_t shared_bytes = 2 * kWavefrontSize * states * sizeof(T);
      if (use_grouped_shared) {
        launch_grouped_scan<T, 1, 4>(p, out, x, dt, b, c, a_log, d, gate,
                                     dt_bias, state, stream);
      } else {
        const dim3 blocks(total);
        const bool use_shared = states > 32;
        if (use_shared) {
          mamba_selective_scan_parallel_kernel<T, 1, true>
              <<<blocks, kWavefrontSize, shared_bytes, stream>>>(
                  out, x, dt, b, c, a_log, d, gate, dt_bias, state, batch,
                  length, channels, states);
        } else {
          mamba_selective_scan_parallel_kernel<T, 1, false>
              <<<blocks, kWavefrontSize, 0, stream>>>(
                  out, x, dt, b, c, a_log, d, gate, dt_bias, state, batch,
                  length, channels, states);
        }
      }
    } else {
      const size_t shared_bytes = 4 * kWavefrontSize * states * sizeof(T);
      if (use_grouped_shared) {
        if (batch >= 4 && channels % 16 == 0) {
          // Small channel sets need more independent blocks.
          if (total <= 1024)
            launch_grouped_scan<T, 2, 8>(p, out, x, dt, b, c, a_log, d, gate,
                                         dt_bias, state, stream);
          else
            launch_grouped_scan<T, 2, 16>(p, out, x, dt, b, c, a_log, d, gate,
                                          dt_bias, state, stream);
        } else {
          launch_grouped_scan<T, 2, 4>(p, out, x, dt, b, c, a_log, d, gate,
                                       dt_bias, state, stream);
        }
      } else {
        const dim3 blocks(total);
        const bool use_shared = states > 32;
        if (use_shared) {
          mamba_selective_scan_parallel_kernel<T, 2, true>
              <<<blocks, kWavefrontSize, shared_bytes, stream>>>(
                  out, x, dt, b, c, a_log, d, gate, dt_bias, state, batch,
                  length, channels, states);
        } else {
          mamba_selective_scan_parallel_kernel<T, 2, false>
              <<<blocks, kWavefrontSize, 0, stream>>>(
                  out, x, dt, b, c, a_log, d, gate, dt_bias, state, batch,
                  length, channels, states);
        }
      }
    }
    return;
  }

  // Distribute long S16 work across more CUs without changing the math.
  const unsigned int threads =
      length > 128 && states == 16 && total <= 4096 ? 64 : 256;
  const dim3 blocks((total + threads - 1) / threads);
  mamba_selective_scan_kernel<T><<<blocks, threads, 0, stream>>>(
      out, x, dt, b, c, a_log, d, gate, dt_bias, state, batch, length, channels,
      states);
}
}  // anonymous namespace
}  // namespace hygon_mamba_scan_detail

struct MambaScanHygonBackend : Runtime<Device::Type::kHygon> {
  static void CheckLaunch() {
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("Hygon MambaSelectiveScan CUDA launch failed");
  }
};

template <>
class Operator<MambaSelectiveScan, Device::Type::kHygon>
    : public CudaMambaSelectiveScan<MambaScanHygonBackend> {
 public:
  using CudaMambaSelectiveScan<MambaScanHygonBackend>::CudaMambaSelectiveScan;

  void operator()(Tensor out, const Tensor x, const Tensor dt, const Tensor b,
                  const Tensor c, const Tensor a_log, const Tensor d,
                  const Tensor gate, const Tensor dt_bias,
                  Tensor state) const override {
    // Optimized paths require contiguous low-precision data and matching
    // parameter dtypes. Other supported layouts/dtypes use the generic path.
    if (layout_.contiguous && a_log.dtype() == dtype_ && d.dtype() == dtype_ &&
        dtype_ != DataType::kFloat32) {
      const auto stream = static_cast<cudaStream_t>(stream_);
      DispatchFunc<List<DataType::kFloat16, DataType::kBFloat16>>(
          dtype_,
          [&](auto tag) {
            using T = TypeMapType<Device::Type::kHygon, decltype(tag)::value>;
            hygon_mamba_scan_detail::launch_scan_dispatch<T>(
                layout_, static_cast<T *>(out.data()),
                static_cast<const T *>(x.data()),
                static_cast<const T *>(dt.data()),
                static_cast<const T *>(b.data()),
                static_cast<const T *>(c.data()),
                static_cast<const T *>(a_log.data()),
                static_cast<const T *>(d.data()),
                static_cast<const T *>(gate.data()),
                static_cast<const T *>(dt_bias.data()),
                static_cast<float *>(state.data()), stream);
          },
          "HygonMambaSelectiveScan");
      if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("Hygon mamba scan launch failed");
      return;
    }
    CudaMambaSelectiveScan<MambaScanHygonBackend>::operator()(
        out, x, dt, b, c, a_log, d, gate, dt_bias, state);
  }

  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_MAMBA_SELECTIVE_SCAN_KERNEL_H_
