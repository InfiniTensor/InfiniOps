// Hygon-specific causal_conv1d kernels migrated from the InfiniCore
// indicator-5 submission (zxk branch, 2026-10-31):
// src/infiniop/ops/causal_conv1d/hygon/causal_conv1d_hygon.cu.
// Decode, tiled prefill, and general indexed/strided paths share the
// InfiniOps operator-specialization interface.
// Copyright 2025 InfiniTensor; MIT notice: LICENSES/InfiniCore-MIT.txt.
#ifndef INFINI_OPS_HYGON_CAUSAL_CONV1D_KERNEL_H_
#define INFINI_OPS_HYGON_CAUSAL_CONV1D_KERNEL_H_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "base/causal_conv1d.h"
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"

namespace infini::ops {
namespace hygon_conv_detail {

struct ConvInfo {
  size_t request_count, T, C, pool_size, total_tokens;
  bool has_bias, has_cu_seqlens, indexed_state_pool;
  int cu_seqlens_kind, initial_indices_kind,
      final_indices_kind;  // 0=none 1=i32 2=i64
  ptrdiff_t out_strides[3]{}, conv_state_strides[3]{}, qkv_strides[3]{};
  ptrdiff_t weight_strides[3]{};
  std::vector<ptrdiff_t> bias_strides;
  std::vector<ptrdiff_t> final_conv_state_strides;
};

// Metadata shared by contiguous K4 launch selection and decode capture.
inline bool has_indexed_k4_layout(const ConvInfo &info) {
  return info.indexed_state_pool && info.T > 0 && info.T <= 128 &&
         info.C == 4096 && info.initial_indices_kind != 2 &&
         info.final_indices_kind == 1 && info.has_bias &&
         info.qkv_strides[0] == static_cast<ptrdiff_t>(info.T * 4096) &&
         info.qkv_strides[1] == 4096 && info.qkv_strides[2] == 1 &&
         info.out_strides[0] == static_cast<ptrdiff_t>(info.T * 4096) &&
         info.out_strides[1] == 4096 && info.out_strides[2] == 1 &&
         info.conv_state_strides[0] == 4096 * 3 &&
         info.conv_state_strides[1] == 3 && info.conv_state_strides[2] == 1 &&
         info.weight_strides[0] == 4 && info.weight_strides[2] == 1 &&
         !info.bias_strides.empty() && info.bias_strides[0] == 1;
}

namespace {
struct __align__(16) half8 { half lane[8]; };

struct __align__(16) bfloat168 { __nv_bfloat16 lane[8]; };

template <typename T>
struct causal_vec16;

template <>
struct causal_vec16<float> {
  using type = float4;
};

template <>
struct causal_vec16<half> {
  using type = half8;
};

template <>
struct causal_vec16<__nv_bfloat16> {
  using type = bfloat168;
};

template <typename T>
__device__ __forceinline__ float load_as_float(const T *ptr, ptrdiff_t offset) {
  return static_cast<float>(ptr[offset]);
}

template <>
__device__ __forceinline__ float load_as_float<half>(const half *ptr,
                                                     ptrdiff_t offset) {
  return __half2float(ptr[offset]);
}

template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(
    const __nv_bfloat16 *ptr, ptrdiff_t offset) {
  return __bfloat162float(ptr[offset]);
}

template <typename T>
__device__ __forceinline__ T cast_from_float(float x) {
  return static_cast<T>(x);
}

template <>
__device__ __forceinline__ half cast_from_float<half>(float x) {
  return __float2half(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16
cast_from_float<__nv_bfloat16>(float x) {
  const uint32_t bits = __float_as_uint(x);
  const uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
  const uint16_t result = (bits & 0x7fffffffu) > 0x7f800000u
                              ? 0x7fffu
                              : static_cast<uint16_t>(rounded >> 16);
  return __ushort_as_bfloat16(result);
}

template <typename T>
__device__ __forceinline__ float conv4_with_bias(float w0, float h0, float w1,
                                                 float h1, float w2, float h2,
                                                 float w3, float h3,
                                                 float bias) {
  float acc = w0 * h0 + w1 * h1 + w2 * h2 + w3 * h3;
  if constexpr (std::is_same_v<T, half>) {
    T rounded = cast_from_float<T>(acc);
    acc = load_as_float(&rounded, 0);
  }
  return acc + bias;
}

template <typename T, int kVec>
__device__ __forceinline__ void load_vec16(const T *ptr, ptrdiff_t offset,
                                           float *values) {
  using Vec = typename causal_vec16<T>::type;
  const Vec value = *reinterpret_cast<const Vec *>(ptr + offset);
  const T *lanes = reinterpret_cast<const T *>(&value);
#pragma unroll
  for (int lane = 0; lane < kVec; ++lane) {
    values[lane] = load_as_float(lanes, lane);
  }
}

template <typename T, int kVec>
__device__ __forceinline__ void store_vec16(T *ptr, ptrdiff_t offset,
                                            const float *values) {
  using Vec = typename causal_vec16<T>::type;
  Vec value;
  T *lanes = reinterpret_cast<T *>(&value);
#pragma unroll
  for (int lane = 0; lane < kVec; ++lane) {
    lanes[lane] = cast_from_float<T>(values[lane]);
  }
  *reinterpret_cast<Vec *>(ptr + offset) = value;
}

__device__ __forceinline__ int64_t load_optional_index(const void *indices,
                                                       bool is_i64, int idx,
                                                       int fallback) {
  if (indices == nullptr) {
    return static_cast<int64_t>(fallback);
  }
  return is_i64
             ? static_cast<const int64_t *>(indices)[idx]
             : static_cast<int64_t>(static_cast<const int32_t *>(indices)[idx]);
}

template <typename T, bool kHasInitialState, int kChannelsPerThread>
__global__
__launch_bounds__(64) void causal_conv1d_k4_lowp_contiguous_t16_kernel(
    T *out, T *conv_state, const T *qkv, const T *weight, const T *bias,
    const int32_t *initial_state_indices, const int32_t *final_state_indices,
    size_t pool_size) {
  constexpr int kChannels = 4096;
  constexpr int kTokens = 16;
  constexpr int kTile = 4;
  constexpr int kStateSize = 3;
  const int request = static_cast<int>(blockIdx.x);
  const int chunk = static_cast<int>(blockIdx.y);
  const int channel_start = static_cast<int>(
      blockIdx.z * blockDim.x * kChannelsPerThread + threadIdx.x);
  const int read_slot =
      kHasInitialState ? initial_state_indices[request] : request;
  const int write_slot = final_state_indices[request];
  if (read_slot < 0 || write_slot < 0 ||
      read_slot >= static_cast<int64_t>(pool_size) ||
      write_slot >= static_cast<int64_t>(pool_size)) {
    return;
  }

#pragma unroll
  for (int channel_offset = 0; channel_offset < kChannelsPerThread;
       ++channel_offset) {
    const int channel = channel_start + channel_offset * blockDim.x;
    if (channel >= kChannels) {
      continue;
    }

    const int weight_offset = channel * 4;
    const float weight0 = load_as_float(weight, weight_offset);
    const float weight1 = load_as_float(weight, weight_offset + 1);
    const float weight2 = load_as_float(weight, weight_offset + 2);
    const float weight3 = load_as_float(weight, weight_offset + 3);
    const float bias_value = load_as_float(bias, channel);
    const ptrdiff_t batch_base =
        static_cast<ptrdiff_t>(request) * kTokens * kChannels + channel;

    if (chunk == 0) {
      const ptrdiff_t read_base =
          (static_cast<ptrdiff_t>(read_slot) * kChannels + channel) *
          kStateSize;
      float history0 = load_as_float(conv_state, read_base);
      float history1 = load_as_float(conv_state, read_base + 1);
      float history2 = load_as_float(conv_state, read_base + 2);
#pragma unroll
      for (int token = 0; token < 3; ++token) {
        const ptrdiff_t offset = batch_base + token * kChannels;
        const float current = load_as_float(qkv, offset);
        const float result =
            conv4_with_bias<T>(weight0, history0, weight1, history1, weight2,
                               history2, weight3, current, bias_value);
        out[offset] = cast_from_float<T>(result);
        history0 = history1;
        history1 = history2;
        history2 = current;
      }

      const ptrdiff_t final_offset = batch_base + 15 * kChannels;
      const float previous0 = load_as_float(qkv, batch_base + 12 * kChannels);
      const float previous1 = load_as_float(qkv, batch_base + 13 * kChannels);
      const float previous2 = load_as_float(qkv, batch_base + 14 * kChannels);
      const float current = load_as_float(qkv, final_offset);
      const float result =
          conv4_with_bias<T>(weight0, previous0, weight1, previous1, weight2,
                             previous2, weight3, current, bias_value);
      out[final_offset] = cast_from_float<T>(result);

      const ptrdiff_t write_base =
          (static_cast<ptrdiff_t>(write_slot) * kChannels + channel) *
          kStateSize;
      conv_state[write_base] = cast_from_float<T>(previous1);
      conv_state[write_base + 1] = cast_from_float<T>(previous2);
      conv_state[write_base + 2] = cast_from_float<T>(current);
    } else {
      const int first_token = 3 + (chunk - 1) * kTile;
      const ptrdiff_t first_offset = batch_base + first_token * kChannels;
      float history0 = load_as_float(qkv, first_offset - 3 * kChannels);
      float history1 = load_as_float(qkv, first_offset - 2 * kChannels);
      float history2 = load_as_float(qkv, first_offset - kChannels);
#pragma unroll
      for (int tile_token = 0; tile_token < kTile; ++tile_token) {
        const ptrdiff_t offset = first_offset + tile_token * kChannels;
        const float current = load_as_float(qkv, offset);
        const float result =
            conv4_with_bias<T>(weight0, history0, weight1, history1, weight2,
                               history2, weight3, current, bias_value);
        out[offset] = cast_from_float<T>(result);
        history0 = history1;
        history1 = history2;
        history2 = current;
      }
    }
  }
}

template <typename T, int kThreads>
__global__ __launch_bounds__(kThreads) void causal_conv1d_k4_decode_kernel(
    T *out, T *conv_state, const T *qkv, const T *weight, const T *bias,
    const int32_t *initial_state_indices, const int32_t *final_state_indices,
    size_t pool_size) {
  constexpr int kChannels = 4096;
  const int request = blockIdx.y;
  const int channel = blockIdx.x * kThreads + threadIdx.x;
  const int read_slot = initial_state_indices == nullptr
                            ? request
                            : initial_state_indices[request];
  const int write_slot = final_state_indices[request];
  if (read_slot < 0 || write_slot < 0 ||
      static_cast<size_t>(read_slot) >= pool_size ||
      static_cast<size_t>(write_slot) >= pool_size) {
    return;
  }
  const ptrdiff_t read_base =
      (static_cast<ptrdiff_t>(read_slot) * kChannels + channel) * 3;
  const ptrdiff_t write_base =
      (static_cast<ptrdiff_t>(write_slot) * kChannels + channel) * 3;
  const T history0 = conv_state[read_base];
  const T history1 = conv_state[read_base + 1];
  const T history2 = conv_state[read_base + 2];
  const ptrdiff_t input_index =
      static_cast<ptrdiff_t>(request) * kChannels + channel;
  const T current = qkv[input_index];
  float weights[4];
  if ((reinterpret_cast<uintptr_t>(weight) & 7) == 0) {
    const uint2 packed = *reinterpret_cast<const uint2 *>(weight + channel * 4);
    const T *values = reinterpret_cast<const T *>(&packed);
#pragma unroll
    for (int index = 0; index < 4; ++index) {
      weights[index] = load_as_float(values, index);
    }
  } else {
#pragma unroll
    for (int index = 0; index < 4; ++index) {
      weights[index] = load_as_float(weight, channel * 4 + index);
    }
  }
  const float result = conv4_with_bias<T>(
      weights[0], load_as_float(&history0, 0), weights[1],
      load_as_float(&history1, 0), weights[2], load_as_float(&history2, 0),
      weights[3], load_as_float(&current, 0), load_as_float(bias, channel));
  out[input_index] = cast_from_float<T>(result);
  conv_state[write_base] = history1;
  conv_state[write_base + 1] = history2;
  conv_state[write_base + 2] = current;
}

template <typename T>
__device__ __forceinline__ float causal_decode_bits(unsigned int bits) {
  const unsigned short raw = static_cast<unsigned short>(bits);
  T value;
  __builtin_memcpy(&value, &raw, sizeof(T));
  return load_as_float<T>(&value, 0);
}

template <typename T>
__device__ __forceinline__ unsigned short causal_encode_bits(float value) {
  const T converted = cast_from_float<T>(value);
  unsigned short bits;
  __builtin_memcpy(&bits, &converted, sizeof(bits));
  return bits;
}

struct alignas(4) CausalHistoryPair {
  unsigned int values[3];
};

template <typename T, int kThreads, bool kSingleRequestPool = false>
__global__ __launch_bounds__(kThreads) void causal_conv1d_k4_decode_pair_kernel(
    T *out, T *conv_state, const T *qkv, const T *weight, const T *bias,
    const int32_t *initial_state_indices, const int32_t *final_state_indices,
    size_t pool_size) {
  constexpr int kChannels = 4096;
  const int request = kSingleRequestPool ? 0 : blockIdx.y;
  const int channel = (blockIdx.x * kThreads + threadIdx.x) * 2;
  const int read_slot = initial_state_indices == nullptr
                            ? request
                            : initial_state_indices[request];
  const int write_slot = final_state_indices[request];
  if constexpr (kSingleRequestPool) {
    if (read_slot != 0 || write_slot != 0) return;
  } else if (read_slot < 0 || write_slot < 0 ||
             static_cast<size_t>(read_slot) >= pool_size ||
             static_cast<size_t>(write_slot) >= pool_size) {
    return;
  }
  const ptrdiff_t read_base =
      (static_cast<ptrdiff_t>(kSingleRequestPool ? 0 : read_slot) * kChannels +
       channel) *
      3;
  const ptrdiff_t write_base =
      (static_cast<ptrdiff_t>(kSingleRequestPool ? 0 : write_slot) * kChannels +
       channel) *
      3;
  const ptrdiff_t input_base =
      static_cast<ptrdiff_t>(request) * kChannels + channel;
  const CausalHistoryPair history =
      *reinterpret_cast<const CausalHistoryPair *>(conv_state + read_base);
  const uint4 weights = *reinterpret_cast<const uint4 *>(weight + channel * 4);
  const unsigned int current =
      *reinterpret_cast<const unsigned int *>(qkv + input_base);
  const unsigned int biases =
      *reinterpret_cast<const unsigned int *>(bias + channel);
  const unsigned int packed_weights[4] = {weights.x, weights.y, weights.z,
                                          weights.w};
  const unsigned int raw_history[6] = {
      history.values[0], history.values[0] >> 16,
      history.values[1], history.values[1] >> 16,
      history.values[2], history.values[2] >> 16};
  unsigned short encoded[2];
#pragma unroll
  for (int offset = 0; offset < 2; ++offset) {
    const unsigned int first = packed_weights[offset * 2];
    const unsigned int second = packed_weights[offset * 2 + 1];
    const float result =
        conv4_with_bias<T>(causal_decode_bits<T>(first),
                           causal_decode_bits<T>(raw_history[offset * 3]),
                           causal_decode_bits<T>(first >> 16),
                           causal_decode_bits<T>(raw_history[offset * 3 + 1]),
                           causal_decode_bits<T>(second),
                           causal_decode_bits<T>(raw_history[offset * 3 + 2]),
                           causal_decode_bits<T>(second >> 16),
                           causal_decode_bits<T>(current >> (offset * 16)),
                           causal_decode_bits<T>(biases >> (offset * 16)));
    encoded[offset] = causal_encode_bits<T>(result);
  }
  *reinterpret_cast<unsigned int *>(out + input_base) =
      static_cast<unsigned int>(encoded[0]) |
      (static_cast<unsigned int>(encoded[1]) << 16);
  CausalHistoryPair updated;
  updated.values[0] = (history.values[0] >> 16) | (history.values[1] << 16);
  updated.values[1] = (current & 0xffffu) | (history.values[2] << 16);
  updated.values[2] = (history.values[2] >> 16) | (current & 0xffff0000u);
  *reinterpret_cast<CausalHistoryPair *>(conv_state + write_base) = updated;
}

template <typename T, int kTimeTile>
__global__ void causal_conv1d_k4_contiguous_tiled_kernel(
    T *out, T *conv_state, const T *qkv, const T *weight, const T *bias,
    const int32_t *initial_state_indices, const int32_t *final_state_indices,
    size_t token_count, size_t pool_size) {
  constexpr int kChannels = 4096;
  const int request = blockIdx.x;
  const int chunk = blockIdx.y;
  const int channel = blockIdx.z * blockDim.x + threadIdx.x;
  const int read_slot = initial_state_indices == nullptr
                            ? request
                            : initial_state_indices[request];
  const int write_slot = final_state_indices[request];
  if (read_slot < 0 || write_slot < 0 ||
      read_slot >= static_cast<int64_t>(pool_size) ||
      write_slot >= static_cast<int64_t>(pool_size)) {
    return;
  }

  const ptrdiff_t weight_offset = static_cast<ptrdiff_t>(channel) * 4;
  const float weight0 = load_as_float(weight, weight_offset);
  const float weight1 = load_as_float(weight, weight_offset + 1);
  const float weight2 = load_as_float(weight, weight_offset + 2);
  const float weight3 = load_as_float(weight, weight_offset + 3);
  const float bias_value = load_as_float(bias, channel);
  const ptrdiff_t batch_base =
      static_cast<ptrdiff_t>(request) * token_count * kChannels + channel;

  if (chunk == 0) {
    const ptrdiff_t read_base =
        (static_cast<ptrdiff_t>(read_slot) * kChannels + channel) * 3;
    float history0 = load_as_float(conv_state, read_base);
    float history1 = load_as_float(conv_state, read_base + 1);
    float history2 = load_as_float(conv_state, read_base + 2);
    for (int token = 0; token < 3; ++token) {
      if (token >= token_count) {
        break;
      }
      const ptrdiff_t offset = batch_base + token * kChannels;
      const float current = load_as_float(qkv, offset);
      const float result =
          conv4_with_bias<T>(weight0, history0, weight1, history1, weight2,
                             history2, weight3, current, bias_value);
      out[offset] = cast_from_float<T>(result);
      history0 = history1;
      history1 = history2;
      history2 = current;
    }

    if (token_count >= 4) {
      const ptrdiff_t final_offset = batch_base + (token_count - 1) * kChannels;
      const float previous0 = load_as_float(qkv, final_offset - 3 * kChannels);
      const float previous1 = load_as_float(qkv, final_offset - 2 * kChannels);
      const float previous2 = load_as_float(qkv, final_offset - kChannels);
      const float current = load_as_float(qkv, final_offset);
      const float result =
          conv4_with_bias<T>(weight0, previous0, weight1, previous1, weight2,
                             previous2, weight3, current, bias_value);
      out[final_offset] = cast_from_float<T>(result);
      history0 = previous1;
      history1 = previous2;
      history2 = current;
    }

    const ptrdiff_t write_base =
        (static_cast<ptrdiff_t>(write_slot) * kChannels + channel) * 3;
    conv_state[write_base] = cast_from_float<T>(history0);
    conv_state[write_base + 1] = cast_from_float<T>(history1);
    conv_state[write_base + 2] = cast_from_float<T>(history2);
  } else {
    const int64_t first_token = 3 + static_cast<int64_t>(chunk - 1) * kTimeTile;
    const ptrdiff_t first_offset = batch_base + first_token * kChannels;
    float history0 = load_as_float(qkv, first_offset - 3 * kChannels);
    float history1 = load_as_float(qkv, first_offset - 2 * kChannels);
    float history2 = load_as_float(qkv, first_offset - kChannels);
#pragma unroll
    for (int tile_token = 0; tile_token < kTimeTile; ++tile_token) {
      const int64_t token = first_token + tile_token;
      if (token < static_cast<int64_t>(token_count) - 1) {
        const ptrdiff_t offset = first_offset + tile_token * kChannels;
        const float current = load_as_float(qkv, offset);
        const float result =
            conv4_with_bias<T>(weight0, history0, weight1, history1, weight2,
                               history2, weight3, current, bias_value);
        out[offset] = cast_from_float<T>(result);
        history0 = history1;
        history1 = history2;
        history2 = current;
      }
    }
  }
}

template <typename T, int kVec, bool kVectorIO, int kTimeTile, int kThreads,
          bool kIndexed>
__global__ __launch_bounds__(256) void causal_conv1d_k4_time_tiled_vec_kernel(
    T *out, T *conv_state, T *final_conv_state, const T *qkv, const T *weight,
    const T *bias, const void *initial_state_indices,
    const void *final_state_indices, bool has_bias,
    bool initial_state_indices_i64, bool final_state_indices_i64,
    size_t T_tokens, size_t C, size_t pool_size, ptrdiff_t out_s0,
    ptrdiff_t out_s1, ptrdiff_t out_s2, ptrdiff_t state_s0, ptrdiff_t state_s1,
    ptrdiff_t state_s2, ptrdiff_t final_s0, ptrdiff_t final_s1,
    ptrdiff_t final_s2, ptrdiff_t qkv_s0, ptrdiff_t qkv_s1, ptrdiff_t qkv_s2,
    ptrdiff_t weight_s0, ptrdiff_t weight_s2, ptrdiff_t bias_s0) {
  constexpr int kChannelsPerBlock = kThreads * kVec;
  constexpr bool kSingleChunk = kIndexed && kTimeTile == 16;

  const int64_t request = static_cast<int64_t>(blockIdx.x);
  const int64_t chunk_start =
      kSingleChunk ? 0 : static_cast<int64_t>(blockIdx.y) * kTimeTile;
  const int64_t token_count = static_cast<int64_t>(T_tokens);
  if (chunk_start >= token_count && !(token_count == 0 && chunk_start == 0)) {
    return;
  }

  const int64_t chunk_end =
      kSingleChunk ? token_count : min(chunk_start + kTimeTile, token_count);
  const int64_t channel_start =
      static_cast<int64_t>(blockIdx.z) * kChannelsPerBlock +
      static_cast<int64_t>(threadIdx.x) * kVec;
  if (channel_start >= static_cast<int64_t>(C)) {
    return;
  }

  int64_t read_slot = request;
  int64_t write_slot = request;
  if constexpr (kIndexed) {
    read_slot = load_optional_index(
        initial_state_indices, initial_state_indices_i64, request, request);
    if (final_state_indices != nullptr) {
      write_slot = load_optional_index(
          final_state_indices, final_state_indices_i64, request, request);
    }
    if (read_slot < 0 || write_slot < 0 ||
        read_slot >= static_cast<int64_t>(pool_size) ||
        (final_state_indices != nullptr &&
         write_slot >= static_cast<int64_t>(pool_size))) {
      return;
    }
  }

  float w0[kVec];
  float w1[kVec];
  float w2[kVec];
  float w3[kVec];
  float bias_value[kVec];
  float history0[kVec];
  float history1[kVec];
  float history2[kVec];

  const ptrdiff_t request_state_base =
      static_cast<ptrdiff_t>(read_slot) * state_s0;
  const ptrdiff_t request_qkv_base = static_cast<ptrdiff_t>(request) * qkv_s0;
  const ptrdiff_t request_out_base = static_cast<ptrdiff_t>(request) * out_s0;

#pragma unroll
  for (int lane = 0; lane < kVec; ++lane) {
    const int64_t channel = channel_start + lane;
    if (channel >= static_cast<int64_t>(C)) {
      w0[lane] = 0.0f;
      w1[lane] = 0.0f;
      w2[lane] = 0.0f;
      w3[lane] = 0.0f;
      bias_value[lane] = 0.0f;
      history0[lane] = 0.0f;
      history1[lane] = 0.0f;
      history2[lane] = 0.0f;
      continue;
    }

    const ptrdiff_t state_base =
        request_state_base + static_cast<ptrdiff_t>(channel) * state_s1;
    const ptrdiff_t weight_base = static_cast<ptrdiff_t>(channel) * weight_s0;
    w0[lane] = load_as_float(weight, weight_base);
    w1[lane] = load_as_float(weight, weight_base + weight_s2);
    w2[lane] = load_as_float(weight, weight_base + 2 * weight_s2);
    w3[lane] = load_as_float(weight, weight_base + 3 * weight_s2);
    bias_value[lane] =
        has_bias
            ? load_as_float(bias, static_cast<ptrdiff_t>(channel) * bias_s0)
            : 0.0f;

    if (chunk_start == 0) {
      history0[lane] = load_as_float(conv_state, state_base);
      history1[lane] = load_as_float(conv_state, state_base + state_s2);
      history2[lane] = load_as_float(conv_state, state_base + 2 * state_s2);
    } else if (chunk_start == 1) {
      history0[lane] = load_as_float(conv_state, state_base + state_s2);
      history1[lane] = load_as_float(conv_state, state_base + 2 * state_s2);
      history2[lane] = load_as_float(
          qkv, request_qkv_base + static_cast<ptrdiff_t>(channel) * qkv_s2);
    } else if (chunk_start == 2) {
      history0[lane] = load_as_float(conv_state, state_base + 2 * state_s2);
      history1[lane] = load_as_float(
          qkv, request_qkv_base + static_cast<ptrdiff_t>(channel) * qkv_s2);
      history2[lane] =
          load_as_float(qkv, request_qkv_base + qkv_s1 +
                                 static_cast<ptrdiff_t>(channel) * qkv_s2);
    } else {
      const ptrdiff_t history_base =
          request_qkv_base + static_cast<ptrdiff_t>(channel) * qkv_s2;
      history0[lane] = load_as_float(
          qkv, history_base + static_cast<ptrdiff_t>(chunk_start - 3) * qkv_s1);
      history1[lane] = load_as_float(
          qkv, history_base + static_cast<ptrdiff_t>(chunk_start - 2) * qkv_s1);
      history2[lane] = load_as_float(
          qkv, history_base + static_cast<ptrdiff_t>(chunk_start - 1) * qkv_s1);
    }
  }

  ptrdiff_t qkv_offset = request_qkv_base + chunk_start * qkv_s1 +
                         static_cast<ptrdiff_t>(channel_start) * qkv_s2;
  ptrdiff_t out_offset = request_out_base + chunk_start * out_s1 +
                         static_cast<ptrdiff_t>(channel_start) * out_s2;

  const bool full_vector =
      channel_start + static_cast<int64_t>(kVec) <= static_cast<int64_t>(C);
  for (int64_t token = chunk_start; token < chunk_end; ++token) {
    float current[kVec];
    if constexpr (kVectorIO) {
      if (full_vector) {
        float result[kVec];
        load_vec16<T, kVec>(qkv, qkv_offset, current);
#pragma unroll
        for (int lane = 0; lane < kVec; ++lane) {
          result[lane] = conv4_with_bias<T>(
              w0[lane], history0[lane], w1[lane], history1[lane], w2[lane],
              history2[lane], w3[lane], current[lane], bias_value[lane]);
        }
        store_vec16<T, kVec>(out, out_offset, result);
      } else {
#pragma unroll
        for (int lane = 0; lane < kVec; ++lane) {
          const int64_t channel = channel_start + lane;
          if (channel < static_cast<int64_t>(C)) {
            current[lane] = load_as_float(
                qkv, qkv_offset + static_cast<ptrdiff_t>(lane) * qkv_s2);
            const float acc = conv4_with_bias<T>(
                w0[lane], history0[lane], w1[lane], history1[lane], w2[lane],
                history2[lane], w3[lane], current[lane], bias_value[lane]);
            out[out_offset + static_cast<ptrdiff_t>(lane) * out_s2] =
                cast_from_float<T>(acc);
          } else {
            current[lane] = 0.0f;
          }
        }
      }
    } else {
#pragma unroll
      for (int lane = 0; lane < kVec; ++lane) {
        const int64_t channel = channel_start + lane;
        if (channel < static_cast<int64_t>(C)) {
          current[lane] = load_as_float(
              qkv, qkv_offset + static_cast<ptrdiff_t>(lane) * qkv_s2);
          const float acc = conv4_with_bias<T>(
              w0[lane], history0[lane], w1[lane], history1[lane], w2[lane],
              history2[lane], w3[lane], current[lane], bias_value[lane]);
          out[out_offset + static_cast<ptrdiff_t>(lane) * out_s2] =
              cast_from_float<T>(acc);
        } else {
          current[lane] = 0.0f;
        }
      }
    }

#pragma unroll
    for (int lane = 0; lane < kVec; ++lane) {
      const int64_t channel = channel_start + lane;
      if (channel < static_cast<int64_t>(C)) {
        history0[lane] = history1[lane];
        history1[lane] = history2[lane];
        history2[lane] = current[lane];
      }
    }
    qkv_offset += qkv_s1;
    out_offset += out_s1;
  }

  if (chunk_end == token_count) {
    T *final_target = final_conv_state;
    ptrdiff_t final_request_base = static_cast<ptrdiff_t>(request) * final_s0;
    ptrdiff_t final_channel_stride = final_s1;
    ptrdiff_t final_tap_stride = final_s2;
    if constexpr (kIndexed) {
      if (final_state_indices != nullptr) {
        final_target = conv_state;
        final_request_base = static_cast<ptrdiff_t>(write_slot) * state_s0;
        final_channel_stride = state_s1;
        final_tap_stride = state_s2;
      }
    }
#pragma unroll
    for (int lane = 0; lane < kVec; ++lane) {
      const int64_t channel = channel_start + lane;
      if (channel < static_cast<int64_t>(C)) {
        const ptrdiff_t final_base =
            final_request_base +
            static_cast<ptrdiff_t>(channel) * final_channel_stride;
        final_target[final_base] = cast_from_float<T>(history0[lane]);
        final_target[final_base + final_tap_stride] =
            cast_from_float<T>(history1[lane]);
        final_target[final_base + 2 * final_tap_stride] =
            cast_from_float<T>(history2[lane]);
      }
    }
  }
}

template <typename T>
__global__ void causal_conv1d_k4_kernel(
    T *out, T *conv_state, T *final_conv_state, const T *qkv, const T *weight,
    const T *bias, const void *cu_seqlens, const void *initial_state_indices,
    const void *final_state_indices, bool has_bias, bool has_cu_seqlens,
    bool cu_seqlens_i64, bool initial_state_indices_i64,
    bool final_state_indices_i64, bool indexed_state_pool, size_t T_tokens,
    size_t C, size_t total_tokens, size_t pool_size, ptrdiff_t out_s0,
    ptrdiff_t out_s1, ptrdiff_t out_s2, ptrdiff_t state_s0, ptrdiff_t state_s1,
    ptrdiff_t state_s2, ptrdiff_t final_s0, ptrdiff_t final_s1,
    ptrdiff_t final_s2, ptrdiff_t qkv_s0, ptrdiff_t qkv_s1, ptrdiff_t qkv_s2,
    ptrdiff_t weight_s0, ptrdiff_t weight_s2, ptrdiff_t bias_s0) {
  const int request = blockIdx.x;
  const int channel = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
  if (channel >= C) {
    return;
  }

  int64_t token_begin = 0;
  int64_t token_end = static_cast<int64_t>(T_tokens);
  int token_batch = request;
  if (has_cu_seqlens) {
    token_begin = load_optional_index(cu_seqlens, cu_seqlens_i64, request, 0);
    token_end = load_optional_index(cu_seqlens, cu_seqlens_i64, request + 1, 0);
    token_batch = 0;
    if (token_begin < 0 || token_end < token_begin ||
        token_end > static_cast<int64_t>(total_tokens)) {
      return;
    }
  }
  const int64_t request_len = token_end - token_begin;

  int64_t read_slot =
      indexed_state_pool
          ? load_optional_index(initial_state_indices,
                                initial_state_indices_i64, request, request)
          : static_cast<int64_t>(request);
  int64_t write_slot =
      indexed_state_pool && final_state_indices != nullptr
          ? load_optional_index(final_state_indices, final_state_indices_i64,
                                request, request)
          : static_cast<int64_t>(request);
  if (read_slot < 0 || write_slot < 0 ||
      read_slot >= static_cast<int64_t>(pool_size) ||
      (final_state_indices != nullptr &&
       write_slot >= static_cast<int64_t>(pool_size))) {
    return;
  }

  const ptrdiff_t state_base = static_cast<ptrdiff_t>(read_slot) * state_s0 +
                               static_cast<ptrdiff_t>(channel) * state_s1;
  const ptrdiff_t weight_base = static_cast<ptrdiff_t>(channel) * weight_s0;
  const float w0 = load_as_float(weight, weight_base);
  const float w1 = load_as_float(weight, weight_base + weight_s2);
  const float w2 = load_as_float(weight, weight_base + 2 * weight_s2);
  const float w3 = load_as_float(weight, weight_base + 3 * weight_s2);
  const float bias_value =
      has_bias ? load_as_float(bias, static_cast<ptrdiff_t>(channel) * bias_s0)
               : 0.0f;

  float history0 = load_as_float(conv_state, state_base);
  float history1 = load_as_float(conv_state, state_base + state_s2);
  float history2 = load_as_float(conv_state, state_base + 2 * state_s2);

  ptrdiff_t qkv_offset = static_cast<ptrdiff_t>(token_batch) * qkv_s0 +
                         static_cast<ptrdiff_t>(token_begin) * qkv_s1 +
                         static_cast<ptrdiff_t>(channel) * qkv_s2;
  ptrdiff_t out_offset = static_cast<ptrdiff_t>(token_batch) * out_s0 +
                         static_cast<ptrdiff_t>(token_begin) * out_s1 +
                         static_cast<ptrdiff_t>(channel) * out_s2;

  for (int64_t t = 0; t < request_len; ++t) {
    const float current = load_as_float(qkv, qkv_offset);
    float acc = conv4_with_bias<T>(w0, history0, w1, history1, w2, history2, w3,
                                   current, bias_value);
    out[out_offset] = cast_from_float<T>(acc);

    history0 = history1;
    history1 = history2;
    history2 = current;
    qkv_offset += qkv_s1;
    out_offset += out_s1;
  }

  T *final_target =
      final_state_indices != nullptr ? conv_state : final_conv_state;
  ptrdiff_t final_base;
  ptrdiff_t final_stride2;
  if (final_state_indices != nullptr) {
    final_base = static_cast<ptrdiff_t>(write_slot) * state_s0 +
                 static_cast<ptrdiff_t>(channel) * state_s1;
    final_stride2 = state_s2;
  } else {
    final_base = static_cast<ptrdiff_t>(request) * final_s0 +
                 static_cast<ptrdiff_t>(channel) * final_s1;
    final_stride2 = final_s2;
  }

  final_target[final_base] = cast_from_float<T>(history0);
  final_target[final_base + final_stride2] = cast_from_float<T>(history1);
  final_target[final_base + 2 * final_stride2] = cast_from_float<T>(history2);
}
}  // anonymous namespace
namespace {

template <typename T, bool kVectorIO, int kTimeTile, int kThreads, int kVec,
          bool kIndexed>
void launch_k4_time_tiled_vec_impl(const ConvInfo &info, void *out,
                                   void *conv_state, void *final_conv_state,
                                   const void *qkv, const void *weight,
                                   const void *bias,
                                   const void *initial_state_indices,
                                   const void *final_state_indices,
                                   cudaStream_t stream) {
  constexpr unsigned int kChannelsPerBlock = kThreads * kVec;

  size_t time_chunks_size = (info.T + kTimeTile - 1) / kTimeTile;
  size_t channel_chunks_size =
      (info.C + kChannelsPerBlock - 1) / kChannelsPerBlock;
  if (time_chunks_size == 0) {
    time_chunks_size = 1;
  }
  if (channel_chunks_size == 0) {
    channel_chunks_size = 1;
  }

  const dim3 grid(static_cast<unsigned int>(info.request_count),
                  static_cast<unsigned int>(time_chunks_size),
                  static_cast<unsigned int>(channel_chunks_size));

  // If the final-state buffer aliases conv_state, the block holding the last
  // token overwrites the history that the first chunk still has to read.
  // Stage through a scratch buffer and copy back once the launch is done.
  T *aliased_scratch = nullptr;
  size_t aliased_elems = 0;
  void *final_state_arg = final_conv_state;
  if (final_conv_state != nullptr && final_conv_state == conv_state &&
      time_chunks_size > 1) {
    const size_t rows = info.pool_size ? info.pool_size : 1;
    aliased_elems =
        (rows - 1) * static_cast<size_t>(info.conv_state_strides[0]) +
        (info.C ? info.C - 1 : 0) *
            static_cast<size_t>(info.conv_state_strides[1]) +
        2 * static_cast<size_t>(info.conv_state_strides[2]) + 1;
    if (cudaMalloc(reinterpret_cast<void **>(&aliased_scratch),
                   aliased_elems * sizeof(T)) == cudaSuccess) {
      final_state_arg = aliased_scratch;
    }
  }

  causal_conv1d_k4_time_tiled_vec_kernel<T, kVec, kVectorIO, kTimeTile,
                                         kThreads, kIndexed>
      <<<grid, kThreads, 0, stream>>>(
          static_cast<T *>(out), static_cast<T *>(conv_state),
          static_cast<T *>(final_state_arg), static_cast<const T *>(qkv),
          static_cast<const T *>(weight), static_cast<const T *>(bias),
          initial_state_indices, final_state_indices, info.has_bias,
          info.initial_indices_kind == 2, info.final_indices_kind == 2, info.T,
          info.C, info.pool_size, info.out_strides[0], info.out_strides[1],
          info.out_strides[2], info.conv_state_strides[0],
          info.conv_state_strides[1], info.conv_state_strides[2],
          info.final_conv_state_strides.empty()
              ? 0
              : info.final_conv_state_strides[0],
          info.final_conv_state_strides.empty()
              ? 0
              : info.final_conv_state_strides[1],
          info.final_conv_state_strides.empty()
              ? 0
              : info.final_conv_state_strides[2],
          info.qkv_strides[0], info.qkv_strides[1], info.qkv_strides[2],
          info.weight_strides[0], info.weight_strides[2],
          info.bias_strides.empty() ? 0 : info.bias_strides[0]);
  if (cudaGetLastError() != cudaSuccess) {
    if (aliased_scratch != nullptr) cudaFree(aliased_scratch);
    throw std::runtime_error("Hygon causal conv1d vec launch failed");
  }
  if (aliased_scratch != nullptr) {
    cudaMemcpyAsync(conv_state, aliased_scratch, aliased_elems * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    cudaFree(aliased_scratch);
  }
}

template <typename T>
bool can_use_k4_vector_io(const ConvInfo &info, const void *out,
                          const void *qkv) {
  constexpr size_t kAlignment = 16;
  const auto is_aligned = [](const void *ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & (kAlignment - 1)) == 0;
  };
  const auto stride_is_aligned = [](ptrdiff_t stride) {
    return (static_cast<size_t>(stride) * sizeof(T) & (kAlignment - 1)) == 0;
  };
  return is_aligned(out) && is_aligned(qkv) &&
         stride_is_aligned(info.out_strides[0]) &&
         stride_is_aligned(info.qkv_strides[0]) &&
         stride_is_aligned(info.out_strides[1]) &&
         stride_is_aligned(info.qkv_strides[1]);
}

template <typename T, int kTimeTile, int kThreads, int kVec, bool kIndexed>
void launch_k4_time_tiled_vec_for_tile(const ConvInfo &info, void *out,
                                       void *conv_state, void *final_conv_state,
                                       const void *qkv, const void *weight,
                                       const void *bias,
                                       const void *initial_state_indices,
                                       const void *final_state_indices,
                                       cudaStream_t stream, bool vector_io) {
  if (vector_io) {
    return launch_k4_time_tiled_vec_impl<T, true, kTimeTile, kThreads, kVec,
                                         kIndexed>(
        info, out, conv_state, final_conv_state, qkv, weight, bias,
        initial_state_indices, final_state_indices, stream);
  }
  return launch_k4_time_tiled_vec_impl<T, false, kTimeTile, kThreads, kVec,
                                       kIndexed>(
      info, out, conv_state, final_conv_state, qkv, weight, bias,
      initial_state_indices, final_state_indices, stream);
}

template <typename T, int kTimeTile, bool kIndexed>
void launch_k4_time_tiled_vec_for_channels(
    const ConvInfo &info, void *out, void *conv_state, void *final_conv_state,
    const void *qkv, const void *weight, const void *bias,
    const void *initial_state_indices, const void *final_state_indices,
    cudaStream_t stream, bool vector_io) {
  if constexpr (kIndexed) {
    return launch_k4_time_tiled_vec_impl<T, false, kTimeTile, 64, 1, true>(
        info, out, conv_state, final_conv_state, qkv, weight, bias,
        initial_state_indices, final_state_indices, stream);
  }
  if (info.C <= 512) {
    constexpr int kSmallVec = sizeof(T) == 4 ? 2 : 4;
    return launch_k4_time_tiled_vec_impl<T, false, kTimeTile, 64, kSmallVec,
                                         kIndexed>(
        info, out, conv_state, final_conv_state, qkv, weight, bias,
        initial_state_indices, final_state_indices, stream);
  }
  constexpr int kFullVec = sizeof(T) == 4 ? 4 : 8;
  if (info.C <= 4096) {
    return launch_k4_time_tiled_vec_for_tile<T, kTimeTile, 128, kFullVec,
                                             kIndexed>(
        info, out, conv_state, final_conv_state, qkv, weight, bias,
        initial_state_indices, final_state_indices, stream, vector_io);
  }
  return launch_k4_time_tiled_vec_for_tile<T, kTimeTile, 256, kFullVec,
                                           kIndexed>(
      info, out, conv_state, final_conv_state, qkv, weight, bias,
      initial_state_indices, final_state_indices, stream, vector_io);
}

template <typename T, bool kIndexed>
void launch_k4_time_tiled_vec(const ConvInfo &info, void *out, void *conv_state,
                              void *final_conv_state, const void *qkv,
                              const void *weight, const void *bias,
                              const void *initial_state_indices,
                              const void *final_state_indices,
                              cudaStream_t stream, bool vector_io) {
  if constexpr (kIndexed) {
    if (info.T == 16) {
      return launch_k4_time_tiled_vec_for_channels<T, 4, true>(
          info, out, conv_state, final_conv_state, qkv, weight, bias,
          initial_state_indices, final_state_indices, stream, vector_io);
    }
  }
  if (info.T <= 16) {
    return launch_k4_time_tiled_vec_for_channels<T, 16, kIndexed>(
        info, out, conv_state, final_conv_state, qkv, weight, bias,
        initial_state_indices, final_state_indices, stream, vector_io);
  }
  if (info.T <= 64) {
    return launch_k4_time_tiled_vec_for_channels<T, 64, kIndexed>(
        info, out, conv_state, final_conv_state, qkv, weight, bias,
        initial_state_indices, final_state_indices, stream, vector_io);
  }
  return launch_k4_time_tiled_vec_for_channels<T, 128, kIndexed>(
      info, out, conv_state, final_conv_state, qkv, weight, bias,
      initial_state_indices, final_state_indices, stream, vector_io);
}

template <typename T, int kTimeTile>
void launch_k4_contiguous_tiled(const ConvInfo &info, void *out,
                                void *conv_state, const void *qkv,
                                const void *weight, const void *bias,
                                const void *initial_state_indices,
                                const void *final_state_indices,
                                cudaStream_t stream) {
  const size_t middle_chunks =
      info.T < 4 ? 0 : (info.T - 4 + kTimeTile - 1) / kTimeTile;
  const dim3 grid(static_cast<unsigned int>(info.request_count),
                  static_cast<unsigned int>(1 + middle_chunks), 64);
  causal_conv1d_k4_contiguous_tiled_kernel<T, kTimeTile>
      <<<grid, 64, 0, stream>>>(
          static_cast<T *>(out), static_cast<T *>(conv_state),
          static_cast<const T *>(qkv), static_cast<const T *>(weight),
          static_cast<const T *>(bias),
          static_cast<const int32_t *>(initial_state_indices),
          static_cast<const int32_t *>(final_state_indices), info.T,
          info.pool_size);
  if (cudaGetLastError() != cudaSuccess)
    throw std::runtime_error("Hygon causal conv1d tiled launch failed");
}

template <typename T, int kChannelsPerThread>
void launch_k4_lowp_t16(const ConvInfo &info, void *out, void *conv_state,
                        const void *qkv, const void *weight, const void *bias,
                        const void *initial_state_indices,
                        const void *final_state_indices, cudaStream_t stream) {
  constexpr unsigned int kThreads = 64;
  const dim3 grid(
      static_cast<unsigned int>(info.request_count), 4,
      (static_cast<unsigned int>(info.C) + kThreads * kChannelsPerThread - 1) /
          (kThreads * kChannelsPerThread));
  if (initial_state_indices == nullptr) {
    causal_conv1d_k4_lowp_contiguous_t16_kernel<T, false, kChannelsPerThread>
        <<<grid, kThreads, 0, stream>>>(
            static_cast<T *>(out), static_cast<T *>(conv_state),
            static_cast<const T *>(qkv), static_cast<const T *>(weight),
            static_cast<const T *>(bias), nullptr,
            static_cast<const int32_t *>(final_state_indices), info.pool_size);
  } else {
    causal_conv1d_k4_lowp_contiguous_t16_kernel<T, true, kChannelsPerThread>
        <<<grid, kThreads, 0, stream>>>(
            static_cast<T *>(out), static_cast<T *>(conv_state),
            static_cast<const T *>(qkv), static_cast<const T *>(weight),
            static_cast<const T *>(bias),
            static_cast<const int32_t *>(initial_state_indices),
            static_cast<const int32_t *>(final_state_indices), info.pool_size);
  }
  if (cudaGetLastError() != cudaSuccess)
    throw std::runtime_error("Hygon causal conv1d T16 launch failed");
}

template <typename T>
void launch_k4(const ConvInfo &info, void *out, void *conv_state,
               void *final_conv_state, const void *qkv, const void *weight,
               const void *bias, const void *cu_seqlens,
               const void *initial_state_indices,
               const void *final_state_indices, cudaStream_t stream) {
  if (!info.has_cu_seqlens && cu_seqlens == nullptr && out != qkv &&
      (final_state_indices != nullptr ||
       !info.final_conv_state_strides.empty()) &&
      info.qkv_strides[2] == 1 && info.out_strides[2] == 1) {
    if (has_indexed_k4_layout(info) &&
        (final_conv_state == nullptr || final_conv_state == conv_state) &&
        final_state_indices != nullptr && bias != nullptr) {
      if constexpr (sizeof(T) == 2) {
        if (info.T == 1 && info.request_count <= 65535) {
          if ((info.request_count == 1 || info.request_count == 4 ||
               info.request_count == 8) &&
              (reinterpret_cast<uintptr_t>(out) & 3) == 0 &&
              (reinterpret_cast<uintptr_t>(conv_state) & 3) == 0 &&
              (reinterpret_cast<uintptr_t>(qkv) & 3) == 0 &&
              (reinterpret_cast<uintptr_t>(bias) & 3) == 0 &&
              (reinterpret_cast<uintptr_t>(weight) & 15) == 0) {
            // A single request with a one-row pool has constant state offsets.
            // Keep device index validation even when the offsets are constant.
            const auto launch_pair = [&](auto single_pool) {
              constexpr int kPairThreads = 128;
              constexpr bool kSingleRequestPool = decltype(single_pool)::value;
              causal_conv1d_k4_decode_pair_kernel<T, kPairThreads,
                                                  kSingleRequestPool>
                  <<<dim3(4096 / (kPairThreads * 2), info.request_count),
                     kPairThreads, 0, stream>>>(
                      static_cast<T *>(out), static_cast<T *>(conv_state),
                      static_cast<const T *>(qkv),
                      static_cast<const T *>(weight),
                      static_cast<const T *>(bias),
                      static_cast<const int32_t *>(initial_state_indices),
                      static_cast<const int32_t *>(final_state_indices),
                      info.pool_size);
            };
            if (info.request_count == 1 && info.pool_size == 1)
              launch_pair(std::true_type{});
            else
              launch_pair(std::false_type{});
            if (cudaGetLastError() != cudaSuccess)
              throw std::runtime_error(
                  "Hygon causal conv1d decode pair launch failed");
            return;
          }
          constexpr int kThreads = 64;
          causal_conv1d_k4_decode_kernel<T, kThreads>
              <<<dim3(4096 / kThreads, info.request_count), kThreads, 0,
                 stream>>>(static_cast<T *>(out), static_cast<T *>(conv_state),
                           static_cast<const T *>(qkv),
                           static_cast<const T *>(weight),
                           static_cast<const T *>(bias),
                           static_cast<const int32_t *>(initial_state_indices),
                           static_cast<const int32_t *>(final_state_indices),
                           info.pool_size);
          if (cudaGetLastError() != cudaSuccess)
            throw std::runtime_error(
                "Hygon causal conv1d decode launch failed");
          return;
        }
        if (info.T == 16) {
          // Small F16 batches favor more independent channel blocks;
          // BF16 B8 benefits from two channels per thread.
          if constexpr (std::is_same_v<T, half>) {
            if (info.request_count == 1)
              return launch_k4_lowp_t16<T, 1>(
                  info, out, conv_state, qkv, weight, bias,
                  initial_state_indices, final_state_indices, stream);
            return launch_k4_lowp_t16<T, 2>(info, out, conv_state, qkv, weight,
                                            bias, initial_state_indices,
                                            final_state_indices, stream);
          } else {
            if (info.request_count == 8)
              return launch_k4_lowp_t16<T, 2>(
                  info, out, conv_state, qkv, weight, bias,
                  initial_state_indices, final_state_indices, stream);
            return launch_k4_lowp_t16<T, 1>(info, out, conv_state, qkv, weight,
                                            bias, initial_state_indices,
                                            final_state_indices, stream);
          }
        }
      }
      if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        if (info.request_count == 4 && info.T == 128)
          return launch_k4_contiguous_tiled<T, 16>(
              info, out, conv_state, qkv, weight, bias, initial_state_indices,
              final_state_indices, stream);
      }
      if (info.T <= 16) {
        return launch_k4_contiguous_tiled<T, 4>(
            info, out, conv_state, qkv, weight, bias, initial_state_indices,
            final_state_indices, stream);
      }
      if (info.request_count == 1 && info.T <= 64) {
        return launch_k4_contiguous_tiled<T, 4>(
            info, out, conv_state, qkv, weight, bias, initial_state_indices,
            final_state_indices, stream);
      }
      if (info.request_count > 1 && info.request_count <= 4 &&
          (info.T < 128 || !std::is_same_v<T, half>)) {
        return launch_k4_contiguous_tiled<T, 8>(
            info, out, conv_state, qkv, weight, bias, initial_state_indices,
            final_state_indices, stream);
      }
      if (info.request_count == 1 && info.T == 128) {
        return launch_k4_contiguous_tiled<T, 8>(
            info, out, conv_state, qkv, weight, bias, initial_state_indices,
            final_state_indices, stream);
      }
      return launch_k4_contiguous_tiled<T, 16>(
          info, out, conv_state, qkv, weight, bias, initial_state_indices,
          final_state_indices, stream);
    }
    const bool vector_io = can_use_k4_vector_io<T>(info, out, qkv);
    if (info.indexed_state_pool) {
      return launch_k4_time_tiled_vec<T, true>(
          info, out, conv_state, final_conv_state, qkv, weight, bias,
          initial_state_indices, final_state_indices, stream, vector_io);
    }
    return launch_k4_time_tiled_vec<T, false>(
        info, out, conv_state, final_conv_state, qkv, weight, bias,
        initial_state_indices, final_state_indices, stream, vector_io);
  }

  constexpr unsigned int threads = 128;
  dim3 grid(static_cast<unsigned int>(info.request_count),
            (static_cast<unsigned int>(info.C) + threads - 1) / threads, 1);
  causal_conv1d_k4_kernel<T><<<grid, threads, 0, stream>>>(
      static_cast<T *>(out), static_cast<T *>(conv_state),
      static_cast<T *>(final_conv_state), static_cast<const T *>(qkv),
      static_cast<const T *>(weight), static_cast<const T *>(bias), cu_seqlens,
      initial_state_indices, final_state_indices, info.has_bias,
      info.has_cu_seqlens, info.cu_seqlens_kind == 2,
      info.initial_indices_kind == 2, info.final_indices_kind == 2,
      info.indexed_state_pool, info.T, info.C, info.total_tokens,
      info.pool_size, info.out_strides[0], info.out_strides[1],
      info.out_strides[2], info.conv_state_strides[0],
      info.conv_state_strides[1], info.conv_state_strides[2],
      info.final_conv_state_strides.empty() ? 0
                                            : info.final_conv_state_strides[0],
      info.final_conv_state_strides.empty() ? 0
                                            : info.final_conv_state_strides[1],
      info.final_conv_state_strides.empty() ? 0
                                            : info.final_conv_state_strides[2],
      info.qkv_strides[0], info.qkv_strides[1], info.qkv_strides[2],
      info.weight_strides[0], info.weight_strides[2],
      info.bias_strides.empty() ? 0 : info.bias_strides[0]);
  if (cudaGetLastError() != cudaSuccess)
    throw std::runtime_error("Hygon causal conv1d launch failed");
}
}  // namespace
}  // namespace hygon_conv_detail

template <>
class Operator<CausalConv1d, Device::Type::kHygon> : public CausalConv1d {
 public:
  Operator(const Tensor qkv, const Tensor weight, Tensor conv_state,
           const std::optional<Tensor> bias,
           const std::optional<Tensor> cu_seqlens,
           const std::optional<Tensor> initial_state_indices,
           const std::optional<Tensor> final_state_indices, Tensor out,
           std::optional<Tensor> final_conv_state)
      : CausalConv1d(qkv, weight, conv_state, bias, cu_seqlens,
                     initial_state_indices, final_state_indices, out,
                     final_conv_state) {
    if (device_.type() != Device::Type::kHygon)
      throw std::invalid_argument("Hygon causal conv requires HYGON device");
    if (weight.shape()[2] != 4 || conv_state.shape()[2] != 3)
      throw std::invalid_argument(
          "Hygon causal conv kernel requires K4 and state_len3");
    check_device();
    if (requests_ >= static_cast<size_t>(INT_MAX) || channels_ > 65535)
      throw std::invalid_argument(
          "Hygon causal conv request/channel launch extent unsupported");
    info_.request_count = requests_;
    info_.T = tokens_;
    info_.C = channels_;
    info_.pool_size = pool_;
    info_.total_tokens = qkv.numel() / (channels_ ? channels_ : 1);
    info_.has_bias = bias.has_value();
    info_.has_cu_seqlens = cu_seqlens.has_value();
    info_.indexed_state_pool =
        final_state_indices.has_value() || final_conv_state.has_value();
    info_.cu_seqlens_kind = index_kind_(cu_seqlens);
    info_.initial_indices_kind = index_kind_(initial_state_indices);
    info_.final_indices_kind = index_kind_(final_state_indices);
    auto fill3 = [&](ptrdiff_t dst[3], const std::optional<Tensor> &t) {
      if (!t) return;
      const auto &s = t->strides();
      for (size_t j = 0; j < s.size() && j < 3; ++j) dst[j] = s[j];
    };
    fill3(info_.out_strides, out);
    fill3(info_.conv_state_strides, conv_state);
    fill3(info_.qkv_strides, qkv);
    if (weight.strides().size() >= 2) {
      info_.weight_strides[0] = weight.strides()[0];
      info_.weight_strides[2] = weight.strides()[2];
    }
    if (bias)
      info_.bias_strides = {static_cast<ptrdiff_t>(
          bias->strides().empty() ? 1 : bias->strides()[0])};
    if (final_conv_state)
      for (auto s : final_conv_state->strides())
        info_.final_conv_state_strides.push_back(s);
    // Only these metadata shapes are advertised for device-graph capture.
    decode_capture_safe_ =
        hygon_conv_detail::has_indexed_k4_layout(info_) &&
        (info_.T == 1 || info_.T == 16 || info_.T == 128) &&
        (info_.request_count == 1 || info_.request_count == 4 ||
         info_.request_count == 8) &&
        (dtype_ == DataType::kFloat16 || dtype_ == DataType::kBFloat16);
  }

  void operator()(const Tensor qkv, const Tensor weight, Tensor conv_state,
                  const std::optional<Tensor> bias,
                  const std::optional<Tensor> cu_seqlens,
                  const std::optional<Tensor> initial_state_indices,
                  const std::optional<Tensor> final_state_indices, Tensor out,
                  std::optional<Tensor> final_conv_state) const override {
    const Views views{qkv,
                      weight,
                      conv_state,
                      bias,
                      cu_seqlens,
                      initial_state_indices,
                      final_state_indices,
                      out,
                      final_conv_state};
    for (size_t i = 0; i < views.size(); ++i) {
      const auto &t = views[i];
      if (t.has_value() != metadata_[i].has_value() ||
          (t && (!std::equal_to<Tensor>{}(*t, *metadata_[i]) ||
                 (t->numel() && !t->data()))))
        throw std::invalid_argument(
            "Hygon causal conv metadata changed or nonempty storage is null");
    }
    check_device();
    if (!requests_ || !channels_) return;
    auto ptr = [&](size_t i) -> void * {
      return views[i] ? const_cast<void *>(views[i]->data()) : nullptr;
    };
    const auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : nullptr);
    switch (dtype_) {
      case DataType::kFloat32:
        hygon_conv_detail::launch_k4<float>(info_, ptr(7), ptr(2), ptr(8),
                                            ptr(0), ptr(1), ptr(3), ptr(4),
                                            ptr(5), ptr(6), stream);
        break;
      case DataType::kFloat16:
        hygon_conv_detail::launch_k4<half>(info_, ptr(7), ptr(2), ptr(8),
                                           ptr(0), ptr(1), ptr(3), ptr(4),
                                           ptr(5), ptr(6), stream);
        break;
      case DataType::kBFloat16:
        hygon_conv_detail::launch_k4<__nv_bfloat16>(
            info_, ptr(7), ptr(2), ptr(8), ptr(0), ptr(1), ptr(3), ptr(4),
            ptr(5), ptr(6), stream);
        break;
      default:
        throw std::invalid_argument("Hygon causal conv dtype unsupported");
    }
  }

  bool capture_safe() const override { return decode_capture_safe_; }

 private:
  hygon_conv_detail::ConvInfo info_{};
  // Canonical indexed K4 kernels launch only GPU work after construction.
  bool decode_capture_safe_ = false;
  static int index_kind_(const std::optional<Tensor> &t) {
    if (!t) return 0;
    return t->dtype() == DataType::kInt64 ? 2 : 1;
  }
  void check_device() const {
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess)
      throw std::runtime_error("Hygon causal conv device query failed");
    if (current != device_.index())
      throw std::invalid_argument("Hygon causal conv owner device mismatch");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_CAUSAL_CONV1D_KERNEL_H_
