#ifndef INFINI_OPS_HYGON_CLOVER_LOCAL_MEMORY_KERNEL_CUH_
#define INFINI_OPS_HYGON_CLOVER_LOCAL_MEMORY_KERNEL_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace infini::ops::hygon_clover_local_memory_detail {

// gfx936 tensor-core kernel for CloverLocalMemory (read-only BD-GDN
// preview), built on the MMAC matrix instructions of the architecture. It
// mirrors the Triton reference at the points that decide the result: the
// normalized q/k rows, the state tile and the softmax weights are rounded
// to the element type before the products, all products accumulate in
// fp32, exp() is the exp2f expansion, and the history term enters the
// value product as its initial accumulator.
//
// One workgroup serves kCols value columns of one (request, value head).
// A wave (64 lanes) owns one 16 x 16 output tile: wave w takes row half
// w & 1 and column tile w >> 1. The two waves of column tile 0 also
// compute their row half's scores and softmax while the state tile is
// being staged; the remaining waves wait that out at the barrier.
//
// The MMAC fragment layouts (verified by a standalone probe):
//   A: lane holds A[lane % 16][(lane / 16) * 4 .. +4),
//   B: from a [col][k] staging the same four consecutive elements of the
//      lane's column,
//   C: lane holds C[lane % 16][4 * r + lane / 16] for r = 0 .. 3.
// Fragments are 8-byte reads of the padded k-contiguous rows; every
// product streams its k steps one 16-wide MMAC at a time.
//
// Grid: x = value_dim / kCols, y = requests, z = value heads. The layout
// requirements (contiguous 16-byte aligned q/k/v rows and state columns,
// element-typed per-head scalars, a power-of-two head sharing) are checked
// by the host; anything else takes the shared SIMT kernel.

constexpr unsigned kWave = 0xffffffffffffffffu;  // one wave: 64 lanes
constexpr int kDim = 128;        // KEY_DIM == VALUE_DIM == 128 (validated)
constexpr int kRows = 32;        // block_size <= 32 (validated)
constexpr int kLd = kDim + 8;    // k-contiguous rows, padded
constexpr int kLdW = kRows + 8;  // key-contiguous rows, padded

using V2I = int __attribute__((ext_vector_type(2)));
using V4F = float __attribute__((ext_vector_type(4)));

template <typename T>
__device__ __forceinline__ V4F Mma(V2I a, V2I b, V4F c) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
  } else {
    return __builtin_amdgcn_mmac_f32_16x16x16f16(a, b, c);
  }
}

// An A or B fragment: the four elements at `base + (lane / 16) * 4`.
template <typename Data>
__device__ __forceinline__ V2I Frag(const Data* __restrict__ base, int lane) {
  const uint16_t* p = reinterpret_cast<const uint16_t*>(base) + 4 * (lane / 16);
  V2I f;
  f.x = static_cast<int>((static_cast<unsigned>(p[1]) << 16) | p[0]);
  f.y = static_cast<int>((static_cast<unsigned>(p[3]) << 16) | p[2]);
  return f;
}

template <typename Data>
__device__ __forceinline__ uint16_t FromFloat(float f) {
  if constexpr (std::is_same_v<Data, __nv_bfloat16>) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(f));
  } else {
    return __half_as_ushort(__float2half_rn(f));
  }
}
template <typename Data>
__device__ __forceinline__ float ToFloat(uint16_t bits) {
  if constexpr (std::is_same_v<Data, __nv_bfloat16>) {
    return __bfloat162float(__ushort_as_bfloat16(bits));
  } else {
    return __half2float(__ushort_as_half(bits));
  }
}
// The per-head scalars, stored in the element type.
template <typename Data>
__device__ __forceinline__ float ReadScalar(const Data* __restrict__ p,
                                            int head) {
  return ToFloat<Data>(reinterpret_cast<const uint16_t*>(p)[head]);
}

// ext_vector_type elements cannot be addressed; a switch keeps the access
// compile-time for the unrolled callers.
__device__ __forceinline__ float Get(V4F v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}
__device__ __forceinline__ float Get(const float4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// The reference lowers exp() to the ex2.approx expansion; exp2f is that.
__device__ __forceinline__ float FastExp(float x) {
  return exp2f(x * 1.4426950408889634f);
}

// Stage eight rounded elements of one padded row.
template <typename Data>
__device__ __forceinline__ void Store8(Data* __restrict__ dst,
                                       const float (&x)[8]) {
  uint16_t* w = reinterpret_cast<uint16_t*>(dst);
#pragma unroll
  for (int e = 0; e < 8; ++e) w[e] = FromFloat<Data>(x[e]);
}

template <typename Data, typename Index, int kCols, int kThreads>
__global__ void __launch_bounds__(kThreads) CloverLocalMemoryMmaKernel(
    const Data* __restrict__ q, const Data* __restrict__ k,
    const Data* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ state, const Index* __restrict__ indices,
    const int8_t* __restrict__ valid, const Data* __restrict__ logit_scale,
    const Data* __restrict__ history_bias, const Data* __restrict__ history_c,
    Data* __restrict__ out, int64_t q_stride_t, int64_t q_stride_h,
    int64_t k_stride_t, int64_t k_stride_h, int64_t v_stride_t,
    int64_t v_stride_h, int64_t g_stride_t, int64_t g_stride_h,
    int64_t state_slot_stride, int64_t state_h_stride, int64_t state_v_stride,
    int64_t out_t_stride, int64_t out_h_stride, int64_t out_v_stride,
    float scale, float reference_block_size, int block_size, int num_slots,
    int q_head_shift, int k_head_shift) {
  constexpr int kTpr = kThreads / kRows;    // staging threads per row
  constexpr int kChunks = kDim / 8 / kTpr;  // 8-element chunks of q/k each
  constexpr int kColTiles = kCols / 16;
  constexpr int kStateElems = kCols * kDim / kThreads;  // state keys staged
  static_assert(kThreads == 2 * kColTiles * 64,
                "one wave per (row half, column tile)");
  static_assert(kStateElems % 8 == 0, "whole 8-key state chunks");

  __shared__ __align__(16) Data q_s[kRows * kLd];   // normalized q rows
  __shared__ __align__(16) Data k_s[kRows * kLd];   // normalized k rows
  __shared__ __align__(16) Data s_s[kCols * kLd];   // state tile, rounded
  __shared__ __align__(16) Data v_s[kCols * kLdW];  // values, [col][key]
  __shared__ __align__(16) Data w_s[kRows * kLdW];  // softmax weights
  __shared__ float history_scale_s[kRows];
  __shared__ int8_t valid_s[kRows];

  const int c0 = static_cast<int>(blockIdx.x) * kCols;
  const int request = static_cast<int>(blockIdx.y);
  const int value_head = static_cast<int>(blockIdx.z);
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t row_base = static_cast<int64_t>(request) * block_size;
  const int wave = tid >> 6;
  const int lane = tid & 63;
  const int row_half = wave & 1;   // the wave's output rows
  const int col_tile = wave >> 1;  // the wave's output columns
  const int r = lane % 16;         // the lane's tile row
  const int half = lane / 16;      // its fragment half
  const int row = tid / kTpr;      // the thread's staging row
  const int part = tid % kTpr;

  const Index cache_index = indices[request];
  const bool request_active = num_slots > 0 && cache_index >= 0 &&
                              cache_index < static_cast<Index>(num_slots);
  Data* const head_out =
      out + row_base * out_t_stride + value_head * out_h_stride;
  if (!request_active) {
    // The reference zeroes every row of an inactive request; rows past the
    // block are not written at all.
    for (int i = tid; i < kRows * kCols; i += kThreads) {
      const int rr = i / kCols;
      if (rr < block_size)
        head_out[rr * out_t_stride + (c0 + i % kCols) * out_v_stride] =
            static_cast<Data>(0.0f);
    }
    return;
  }
  const int64_t slot = cache_index;
  const int q_head = value_head >> q_head_shift;
  const int k_head = value_head >> k_head_shift;
  const bool row_live = row < block_size;

  // ---- stage the normalized q and k rows and the transposed values ----
  {
    const Data* q_src = q + row_base * q_stride_t + q_head * q_stride_h;
    const Data* k_src = k + row_base * k_stride_t + k_head * k_stride_h;
    float sumsq_q = 0.0f, sumsq_k = 0.0f;
    float xq[kChunks][8], xk[kChunks][8];
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      const uint4 pq =
          row_live ? *reinterpret_cast<const uint4*>(q_src + row * q_stride_t +
                                                     8 * (c * kTpr + part))
                   : make_uint4(0, 0, 0, 0);
      const uint4 pk =
          row_live ? *reinterpret_cast<const uint4*>(k_src + row * k_stride_t +
                                                     8 * (c * kTpr + part))
                   : make_uint4(0, 0, 0, 0);
      const uint16_t* wq = reinterpret_cast<const uint16_t*>(&pq);
      const uint16_t* wk = reinterpret_cast<const uint16_t*>(&pk);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        xq[c][e] = ToFloat<Data>(wq[e]);
        xk[c][e] = ToFloat<Data>(wk[e]);
        sumsq_q += xq[c][e] * xq[c][e];
        sumsq_k += xk[c][e] * xk[c][e];
      }
    }
#pragma unroll
    for (int off = 1; off < kTpr; off <<= 1) {
      sumsq_q += __shfl_xor_sync(kWave, sumsq_q, off);
      sumsq_k += __shfl_xor_sync(kWave, sumsq_k, off);
    }
    const float norm_q = rsqrtf(sumsq_q + 1.0e-6f);
    const float norm_k = rsqrtf(sumsq_k + 1.0e-6f);
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      float rq[8], rk[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        rq[e] = xq[c][e] * norm_q;
        rk[e] = xk[c][e] * norm_k;
      }
      Store8<Data>(q_s + row * kLd + 8 * (c * kTpr + part), rq);
      Store8<Data>(k_s + row * kLd + 8 * (c * kTpr + part), rk);
    }
    // The values transposed to [col][key]; one 8-element chunk per thread.
    const int vpart = tid % (kCols / 8);
    const int vrow = tid / (kCols / 8);
    if (vrow < kRows) {
      const uint4 pv = vrow < block_size
                           ? *reinterpret_cast<const uint4*>(
                                 v + (row_base + vrow) * v_stride_t +
                                 value_head * v_stride_h + c0 + 8 * vpart)
                           : make_uint4(0, 0, 0, 0);
      const uint16_t* wv = reinterpret_cast<const uint16_t*>(&pv);
#pragma unroll
      for (int e = 0; e < 8; ++e)
        reinterpret_cast<uint16_t*>(v_s + (8 * vpart + e) * kLdW + vrow)[0] =
            wv[e];
    }
  }
  if (tid < kRows) valid_s[tid] = tid < block_size ? valid[row_base + tid] : 0;

  // The state tile in fp32 registers, converted after the barrier.
  const int column = tid % kCols;
  const int kgroup = tid / kCols;
  float4 s_raw[kStateElems / 4];
  {
    const float* s_src = state + slot * state_slot_stride +
                         value_head * state_h_stride +
                         (c0 + column) * state_v_stride + kStateElems * kgroup;
#pragma unroll
    for (int c = 0; c < kStateElems / 4; ++c)
      s_raw[c] = *reinterpret_cast<const float4*>(s_src + 4 * c);
  }
  // The count is also the barrier the staging above needs; every thread
  // computes the history logit for itself, so no lane reads what another
  // lane of its wave is still writing (gfx936's __syncwarp does not order
  // shared memory).
  const float history_logit =
      ReadScalar<Data>(history_bias, value_head) +
      ReadScalar<Data>(history_c, value_head) *
          logf(fmaxf(static_cast<float>(__syncthreads_count(
                         tid < block_size && valid[row_base + tid] != 0)),
                     1.0f) /
               reference_block_size);

  // ---- the rounded state tile ----
  {
    Data* dst = s_s + column * kLd + kStateElems * kgroup;
#pragma unroll
    for (int c = 0; c < kStateElems / 8; ++c) {
      float x[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        x[e] = Get(s_raw[2 * c], e);
        x[4 + e] = Get(s_raw[2 * c + 1], e);
      }
      Store8<Data>(dst + 8 * c, x);
    }
  }
  // ---- scores and softmax: the column tile 0 waves, once per row half ----
  if (col_tile == 0) {
    const float ls = ReadScalar<Data>(logit_scale, value_head);
    V4F score[2] = {};  // [key tile], lane's C[r][4 * e + half]
#pragma unroll
    for (int step = 0; step < kDim / 16; ++step) {
      const V2I a =
          Frag<Data>(q_s + (16 * row_half + r) * kLd + 16 * step, lane);
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        const V2I b = Frag<Data>(k_s + (16 * kt + r) * kLd + 16 * step, lane);
        score[kt] = Mma<Data>(a, b, score[kt]);
      }
    }
    // A row's 32 logits sit in the four lanes holding it, eight each.
    float logit[2][4];  // [key tile][register]
    float row_max = -INFINITY;
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key = 16 * kt + 4 * e + half;
        const float value = Get(score[kt], e);
        const bool live = key < block_size && valid_s[key] != 0;
        logit[kt][e] = live ? value * ls : -INFINITY;
        row_max = fmaxf(row_max, logit[kt][e]);
      }
    }
    row_max = fmaxf(row_max, __shfl_xor_sync(kWave, row_max, 16));
    row_max = fmaxf(row_max, __shfl_xor_sync(kWave, row_max, 32));
    row_max = fmaxf(row_max, history_logit);
    float local_sum = 0.0f;
#pragma unroll
    for (int kt = 0; kt < 2; ++kt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        logit[kt][e] = FastExp(logit[kt][e] - row_max);
        local_sum += logit[kt][e];
      }
    local_sum += __shfl_xor_sync(kWave, local_sum, 16);
    local_sum += __shfl_xor_sync(kWave, local_sum, 32);
    const float exp_history = FastExp(history_logit - row_max);
    const float inverse = 1.0f / (local_sum + exp_history);
    // The reference rounds the local weights to the element type.
    Data* w_row = w_s + (16 * row_half + r) * kLdW;
#pragma unroll
    for (int kt = 0; kt < 2; ++kt)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        reinterpret_cast<uint16_t*>(w_row + 16 * kt + 4 * e + half)[0] =
            FromFloat<Data>(logit[kt][e] * inverse);
    const float gate =
        scale * FastExp(g[(row_base + 16 * row_half + r) * g_stride_t +
                          value_head * g_stride_h]);
    history_scale_s[16 * row_half + r] = exp_history * inverse * gate;
  }
  __syncthreads();

  // ---- history = q . state, then the values on top of it ----
  V4F result = {};
  {
    const Data* q_row = q_s + (16 * row_half + r) * kLd;
    const Data* s_col = s_s + (16 * col_tile + r) * kLd;
#pragma unroll
    for (int step = 0; step < kDim / 16; ++step)
      result = Mma<Data>(Frag<Data>(q_row + 16 * step, lane),
                         Frag<Data>(s_col + 16 * step, lane), result);
  }
  const float gain = history_scale_s[16 * row_half + r];
  result *= gain;
  {
    const Data* w_row = w_s + (16 * row_half + r) * kLdW;
    const Data* v_col = v_s + (16 * col_tile + r) * kLdW;
#pragma unroll
    for (int step = 0; step < kRows / 16; ++step)
      result = Mma<Data>(Frag<Data>(w_row + 16 * step, lane),
                         Frag<Data>(v_col + 16 * step, lane), result);
  }

  // ---- write-back ----
  const int row_abs = 16 * row_half + r;
  if (row_abs < block_size) {
    const bool keep = valid_s[row_abs] != 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float value = keep ? Get(result, e) : 0.0f;
      Data* dst = head_out + row_abs * out_t_stride +
                  (c0 + 16 * col_tile + 4 * e + half) * out_v_stride;
      reinterpret_cast<uint16_t*>(dst)[0] = FromFloat<Data>(value);
    }
  }
}

}  // namespace infini::ops::hygon_clover_local_memory_detail

#endif  // INFINI_OPS_HYGON_CLOVER_LOCAL_MEMORY_KERNEL_CUH_
