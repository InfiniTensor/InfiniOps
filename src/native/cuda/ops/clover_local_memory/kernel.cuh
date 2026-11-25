#ifndef INFINI_OPS_CUDA_CLOVER_LOCAL_MEMORY_KERNEL_CUH_
#define INFINI_OPS_CUDA_CLOVER_LOCAL_MEMORY_KERNEL_CUH_

#include <cstdint>

#include "data_type.h"
#include "native/cuda/caster.cuh"

namespace infini::ops {

namespace clover_local_memory_detail {

// The logical warp is always 32 lanes, aligned to a 32-thread slice of the
// block. Every warp intrinsic below uses offsets below 32 and an explicit
// width of 32, so the exchange never leaves that slice on warp-32 and
// warp-64 devices alike.
constexpr unsigned kFull = 0xffffffffu;
constexpr int kWarpLanes = 32;

constexpr int kDim = 128;          // KEY_DIM == VALUE_DIM == 128 (validated)
constexpr int kMaxRows = 32;       // block_size <= 32 (validated)
constexpr int kRowsPerBlock = 16;  // query rows per block, one warp each
constexpr int kWarps = kRowsPerBlock;
constexpr int kTileV = 64;         // value columns per block
constexpr int kHalf = kDim / 2;    // key elements staged at a time
// Padded fp32 row strides in words. A stride of 4 (mod 32) maps consecutive
// rows to distinct 4-bank groups, so a "one lane per row" float4 load is
// conflict free, and every stride keeps rows 16-byte aligned. The value tile
// uses an odd word stride so its per-lane float2 reads stay conflict free.
constexpr int kStride = kHalf + 4;        // 68: one fp32 element per key
constexpr int kPairStride = kDim + 4;     // 132: two interleaved columns per key
constexpr int kValStride = kTileV + 2;    // 66

struct Float2 {
  float x, y;
};
struct Float4 {
  float x, y, z, w;
};

__device__ __forceinline__ Float4 Load4(const float* __restrict__ p) {
  return *reinterpret_cast<const Float4*>(p);
}
__device__ __forceinline__ void Store4(float* p, Float4 v) {
  *reinterpret_cast<Float4*>(p) = v;
}
__device__ __forceinline__ Float2 Load2(const float* __restrict__ p) {
  return *reinterpret_cast<const Float2*>(p);
}
__device__ __forceinline__ void Store2(float* p, Float2 v) {
  *reinterpret_cast<Float2*>(p) = v;
}

__device__ __forceinline__ float WarpMax(float x) {
#pragma unroll
  for (int off = kWarpLanes / 2; off > 0; off >>= 1)
    x = fmaxf(x, __shfl_xor_sync(kFull, x, off, kWarpLanes));
  return x;
}

// The reference lowers exp() to the ex2.approx.f32 hardware approximation.
// exp2f() compiles to that same instruction, which keeps the softmax weights
// and the gate within an ulp of the reference's.
__device__ __forceinline__ float FastExp(float x) {
  return exp2f(x * 1.4426950408889634f);
}

__device__ __forceinline__ float WarpSum(float x) {
#pragma unroll
  for (int off = kWarpLanes / 2; off > 0; off >>= 1)
    x += __shfl_xor_sync(kFull, x, off, kWarpLanes);
  return x;
}

// Rounds an fp32 value to the element type and back, the rounding the Triton
// reference applies before its dot products.
template <Device::Type kDevice, typename Data>
__device__ __forceinline__ float Rounded(float x) {
  return Caster<kDevice>::template Cast<float>(
      Caster<kDevice>::template Cast<Data>(x));
}

// Per-head scalars may be fp32 or either reduced type (see the base contract).
template <Device::Type kDevice>
__device__ __forceinline__ float LoadScalar(const void* ptr, int dtype,
                                            int head) {
  if (dtype == static_cast<int>(DataType::kFloat32))
    return reinterpret_cast<const float*>(ptr)[head];
  if (dtype == static_cast<int>(DataType::kFloat16))
    return Caster<kDevice>::template Cast<float>(
        reinterpret_cast<const TypeMapType<kDevice, DataType::kFloat16>*>(
            ptr)[head]);
  return Caster<kDevice>::template Cast<float>(
      reinterpret_cast<const TypeMapType<kDevice, DataType::kBFloat16>*>(
          ptr)[head]);
}

// Reciprocal L2 norm of one 128-wide row held by a warp (4 elements per
// lane), with the GDN epsilon.
template <Device::Type kDevice, typename Data>
__device__ __forceinline__ float RowNorm(const Data* __restrict__ row,
                                         int64_t stride_k, int lane) {
  float sumsq = 0.0f;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float x = Caster<kDevice>::template Cast<float>(
        row[static_cast<int64_t>(lane * 4 + e) * stride_k]);
    sumsq += x * x;
  }
  return rsqrtf(WarpSum(sumsq) + 1.0e-6f);
}

// The i-th element of a q/k row, scaled by its reciprocal norm and rounded
// to the element type like the reference.
template <Device::Type kDevice, typename Data>
__device__ __forceinline__ float Normalized(const Data* __restrict__ row,
                                            int64_t stride_k, int i,
                                            float norm) {
  return Rounded<kDevice, Data>(
      Caster<kDevice>::template Cast<float>(
          row[static_cast<int64_t>(i) * stride_k]) *
      norm);
}

}  // namespace clover_local_memory_detail

// Shared-layer SIMT kernel for CloverLocalMemory, mirroring the Triton
// reference (read-only BD-GDN preview) including its rounding points.
//
// One thread block serves 16 query rows of one (request, value head, 64-wide
// value tile); one warp owns one query row. Inside a warp every lane plays
// two roles at once: lane j is local key j and lane v is the pair of value
// columns (2v, 2v+1). Each lane therefore holds whole dot products and no
// reduction is needed for them:
//   score[t][j]    = q[t] . k[j]
//   history[t][c]  = q[t] . state[c]
//   local[t][c]    = sum_j w[t][j] * values[j][c]
// The operands are staged in shared memory as fp32, so the key loop is pure
// float4 loads and FMAs. Like the reference, the normalized q/k rows and the
// state tile are rounded to the element type before the products, and the
// softmax weights are rounded before the value dot. The key axis is staged
// in two halves so the fp32 tiles fit in shared memory. All reductions have
// a fixed order and the kernel uses no atomics, so it is deterministic.
//
// Grid: x = value tiles (VALUE_DIM / 64), y = requests * row tiles,
//       z = value heads.  Block: 16 warps.
template <Device::Type kDevice, typename Data, typename Index>
// The launch bounds matter on gfx936 (DTK): a 512-thread block this size
// without them is compiled with a 256-thread cap, and the threads past the
// cap then run with wrapped indices and silently corrupt their rows.
__global__ void __launch_bounds__(clover_local_memory_detail::kWarps *
                                  clover_local_memory_detail::kWarpLanes)
    CloverLocalMemoryKernel(
    const Data* __restrict__ q, const Data* __restrict__ k,
    const Data* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ state, const Index* __restrict__ indices,
    const int8_t* __restrict__ valid, const void* __restrict__ logit_scale,
    const void* __restrict__ history_bias_in,
    const void* __restrict__ history_c_in, Data* __restrict__ out,
    int64_t q_stride_t, int64_t q_stride_h, int64_t q_stride_k,
    int64_t k_stride_t, int64_t k_stride_h, int64_t k_stride_k,
    int64_t v_stride_t, int64_t v_stride_h, int64_t v_stride_v,
    int64_t g_stride_t, int64_t g_stride_h, int64_t state_slot_stride,
    int64_t state_h_stride, int64_t state_v_stride, int64_t state_k_stride,
    int64_t out_t_stride, int64_t out_h_stride, int64_t out_v_stride,
    float scale, float reference_block_size, int block_size, int num_slots,
    int num_q_heads, int num_k_heads, int num_v_heads, int logit_scale_dtype,
    int history_bias_dtype, int history_c_dtype) {
  using namespace clover_local_memory_detail;

  __shared__ __align__(16) float q_s[kRowsPerBlock * kStride];  // normalized q
  __shared__ __align__(16) float k_s[kMaxRows * kStride];       // normalized k
  // st_s[pair][2i], st_s[pair][2i + 1]: columns (2*pair, 2*pair + 1) at key i.
  __shared__ __align__(16) float st_s[kTileV / 2 * kPairStride];
  // vs_s[key][2c], vs_s[key][2c + 1]: values of columns (2c, 2c + 1).
  __shared__ __align__(16) float vs_s[kMaxRows * kValStride];
  __shared__ float norm_q[kRowsPerBlock];
  __shared__ float norm_k[kMaxRows];
  __shared__ int8_t valid_s[kMaxRows];

  const int v0 = static_cast<int>(blockIdx.x) * kTileV;
  const int row_tiles = (block_size + kRowsPerBlock - 1) / kRowsPerBlock;
  const int request = static_cast<int>(blockIdx.y) / row_tiles;
  const int row_tile = static_cast<int>(blockIdx.y) % row_tiles;
  const int value_head = static_cast<int>(blockIdx.z);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kWarpLanes;
  const int warp = tid / kWarpLanes;
  const int row = row_tile * kRowsPerBlock + warp;  // this warp's query row
  const int64_t row_base = static_cast<int64_t>(request) * block_size;

  const Index cache_index = indices[request];
  const bool request_active =
      cache_index >= 0 && cache_index < static_cast<Index>(num_slots);
  Data* const head_out =
      out + static_cast<int64_t>(request) * block_size * out_t_stride +
      value_head * out_h_stride;
  if (!request_active) {
    // The reference stores zeros for every row of an inactive request.
    const int rows = min(kRowsPerBlock, block_size - row_tile * kRowsPerBlock);
    for (int i = tid; i < rows * kTileV; i += kWarps * kWarpLanes)
      head_out[static_cast<int64_t>(row_tile * kRowsPerBlock + i / kTileV) *
                   out_t_stride +
               static_cast<int64_t>(v0 + i % kTileV) * out_v_stride] = Data(0);
    return;
  }
  const int64_t safe_slot = cache_index;

  // Value heads may share q/k heads.
  const int q_head = value_head / (num_v_heads / num_q_heads);
  const int k_head = value_head / (num_v_heads / num_k_heads);
  const bool row_live = row < block_size;

  // ---- reciprocal norms of this block's q rows and of the k rows ----
  norm_q[warp] =
      row_live ? RowNorm<kDevice, Data>(
                     q + (row_base + row) * q_stride_t + q_head * q_stride_h,
                     q_stride_k, lane)
               : 0.0f;
  {
    const Data* k_rows = k + row_base * k_stride_t + k_head * k_stride_h;
#pragma unroll
    for (int r = warp; r < kMaxRows; r += kWarps)
      norm_k[r] = r < block_size
                      ? RowNorm<kDevice, Data>(
                            k_rows + static_cast<int64_t>(r) * k_stride_t,
                            k_stride_k, lane)
                      : 0.0f;
  }
  if (tid < block_size) valid_s[tid] = valid[row_base + tid];

  // ---- value tile ----
#pragma unroll
  for (int i = 0; i < kMaxRows / kWarps; ++i) {
    const int j = warp + i * kWarps;  // key row
    Float2 value{0.0f, 0.0f};
    if (j < block_size) {
      const Data* src = v + (row_base + j) * v_stride_t +
                        value_head * v_stride_h +
                        static_cast<int64_t>(v0 + 2 * lane) * v_stride_v;
      value = Float2{
          Caster<kDevice>::template Cast<float>(src[0]),
          Caster<kDevice>::template Cast<float>(
              src[static_cast<int64_t>(1) * v_stride_v])};
    }
    Store2(vs_s + j * kValStride + 2 * lane, value);
  }
  // Everything staged above is read across warps below.
  __syncthreads();

  // ---- per-head scalars and the history-bucket logit ----
  const float ls =
      LoadScalar<kDevice>(logit_scale, logit_scale_dtype, value_head);
  // Lane j is local key j. Invalid keys and lanes past the block carry no
  // weight and are not counted in the valid block size.
  const bool key_live = lane < block_size && valid_s[lane] != 0;
  const float valid_block_size =
      fmaxf(static_cast<float>(static_cast<int>(
                WarpSum(key_live ? 1.0f : 0.0f))),
            1.0f);
  const float hb =
      LoadScalar<kDevice>(history_bias_in, history_bias_dtype, value_head) +
      LoadScalar<kDevice>(history_c_in, history_c_dtype, value_head) *
          logf(valid_block_size / reference_block_size);

  // ---- key axis in two halves: stage, then every warp sweeps its row ----
  // Two independent accumulator chains per dot keep the FMA chains short.
  float score[2] = {0.0f, 0.0f};
  float history_0[2] = {0.0f, 0.0f};
  float history_1[2] = {0.0f, 0.0f};
  // Row bases in words; each load below covers four consecutive keys.
  const float* q4 = q_s + warp * kStride;
  const float* k4 = k_s + lane * kStride;
  const float* s4 = st_s + lane * kPairStride;
  for (int half = 0; half < kDim / kHalf; ++half) {
    // q: 16 rows x kHalf elements, two per thread.
    {
      const int t = tid / (kHalf / 2);
      const int c = (tid % (kHalf / 2)) * 2;
      const int r = row_tile * kRowsPerBlock + t;
      Float2 value{0.0f, 0.0f};
      if (r < block_size) {
        const Data* src = q + (row_base + r) * q_stride_t +
                          q_head * q_stride_h +
                          static_cast<int64_t>(half * kHalf + c) * q_stride_k;
        value = Float2{Normalized<kDevice, Data>(src, q_stride_k, 0, norm_q[t]),
                       Normalized<kDevice, Data>(src, q_stride_k, 1, norm_q[t])};
      }
      Store2(q_s + t * kStride + c, value);
    }
    // k: 32 rows x kHalf elements, four per thread.
    {
      const int r = tid / (kHalf / 4);
      const int c = (tid % (kHalf / 4)) * 4;
      Float4 value{0.0f, 0.0f, 0.0f, 0.0f};
      if (r < block_size) {
        const Data* src = k + (row_base + r) * k_stride_t +
                          k_head * k_stride_h +
                          static_cast<int64_t>(half * kHalf + c) * k_stride_k;
        value = Float4{Normalized<kDevice, Data>(src, k_stride_k, 0, norm_k[r]),
                       Normalized<kDevice, Data>(src, k_stride_k, 1, norm_k[r]),
                       Normalized<kDevice, Data>(src, k_stride_k, 2, norm_k[r]),
                       Normalized<kDevice, Data>(src, k_stride_k, 3, norm_k[r])};
      }
      Store4(k_s + r * kStride + c, value);
    }
    // state: 32 column pairs x kHalf keys, four keys per thread.
    {
      const int pair = tid / (kHalf / 4);
      const int c = (tid % (kHalf / 4)) * 4;
      const float* col0 = state + safe_slot * state_slot_stride +
                          value_head * state_h_stride +
                          static_cast<int64_t>(v0 + 2 * pair) * state_v_stride +
                          static_cast<int64_t>(half * kHalf + c) * state_k_stride;
      const float* col1 = col0 + state_v_stride;
      float* dst = st_s + pair * kPairStride + 2 * c;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        // The reference casts the fp32 state tile to the element type first.
        Store2(dst + 2 * e,
               Float2{Rounded<kDevice, Data>(
                          col0[static_cast<int64_t>(e) * state_k_stride]),
                      Rounded<kDevice, Data>(
                          col1[static_cast<int64_t>(e) * state_k_stride])});
      }
    }
    __syncthreads();

    // One sweep over this key half. q is a broadcast read that feeds the
    // score against key j and the history of columns (2v, 2v+1) at once.
    if (row_live) {
#pragma unroll
      for (int i = 0; i < kHalf / 8; ++i) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const int n = 2 * i + u;  // group of four keys
          const Float4 a = Load4(q4 + 4 * n);      // q at keys 4n .. 4n + 3
          const Float4 b = Load4(k4 + 4 * n);      // k at keys 4n .. 4n + 3
          const Float4 sa = Load4(s4 + 8 * n);     // keys 4n, 4n + 1
          const Float4 sb = Load4(s4 + 8 * n + 4);  // keys 4n + 2, 4n + 3
          score[u] += a.x * b.x;
          score[u] += a.y * b.y;
          score[u] += a.z * b.z;
          score[u] += a.w * b.w;
          history_0[u] += a.x * sa.x;
          history_0[u] += a.y * sa.z;
          history_0[u] += a.z * sb.x;
          history_0[u] += a.w * sb.z;
          history_1[u] += a.x * sa.y;
          history_1[u] += a.y * sa.w;
          history_1[u] += a.z * sb.y;
          history_1[u] += a.w * sb.w;
        }
      }
    }
    __syncthreads();  // the tiles are restaged for the next half
  }
  if (!row_live) return;

  // ---- softmax over the local keys plus the history bucket ----
  // Invalid keys and lanes past the block contribute -inf, a zero weight.
  const float logit =
      key_live ? (score[0] + score[1]) * ls : -INFINITY;
  const float row_max = fmaxf(WarpMax(logit), hb);
  const float exp_history = FastExp(hb - row_max);
  const float exp_local = FastExp(logit - row_max);
  const float denominator = WarpSum(exp_local) + exp_history;
  // The reference rounds the local weights to the element type before its
  // value dot product.
  const float weight = Rounded<kDevice, Data>(exp_local / denominator);
  const float history_weight = exp_history / denominator;

  // ---- value dot: lane v gathers every key's weight and its two values ----
  float local[2] = {0.0f, 0.0f};
  const float* col = vs_s + 2 * lane;
  for (int j = 0; j < block_size; ++j) {
    const float wj = __shfl_sync(kFull, weight, j, kWarpLanes);
    const Float2 value = Load2(col + j * kValStride);
    local[0] += wj * value.x;
    local[1] += wj * value.y;
  }

  // out = history * (scale * exp(g) * history_weight) + local
  const float gate =
      scale * FastExp(g[(row_base + row) * g_stride_t + value_head * g_stride_h]);
  const float gain = gate * history_weight;
  const bool keep = valid_s[row] != 0;
  const Data result[2] = {
      keep ? Caster<kDevice>::template Cast<Data>(
                 (history_0[0] + history_0[1]) * gain + local[0])
           : Data(0),
      keep ? Caster<kDevice>::template Cast<Data>(
                 (history_1[0] + history_1[1]) * gain + local[1])
           : Data(0)};
  Data* const dst = head_out + static_cast<int64_t>(row) * out_t_stride +
                    static_cast<int64_t>(v0 + 2 * lane) * out_v_stride;
  if (out_v_stride == 1 &&
      (reinterpret_cast<uintptr_t>(dst) & 3) == 0) {
    // Adjacent columns: one 32-bit store for the coalesced pair.
    *reinterpret_cast<uint32_t*>(dst) =
        *reinterpret_cast<const uint32_t*>(result);
  } else {
    dst[0] = result[0];
    dst[out_v_stride] = result[1];
  }
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_LOCAL_MEMORY_KERNEL_CUH_
