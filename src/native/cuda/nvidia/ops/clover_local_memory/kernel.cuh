#ifndef INFINI_OPS_NVIDIA_CLOVER_LOCAL_MEMORY_KERNEL_CUH_
#define INFINI_OPS_NVIDIA_CLOVER_LOCAL_MEMORY_KERNEL_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace infini::ops::nvidia_clover_local_memory_detail {

// NVIDIA tensor-core kernel for CloverLocalMemory (read-only BD-GDN
// preview), for sm_80 and later. It mirrors the Triton reference operation
// for operation, down to the instructions that decide the result:
//   - the three products are the same bf16/f16 MMA (m16n8k16, fp32
//     accumulation, K-steps in order) that tl.dot() lowers to,
//   - normalized q/k rows, the state tile and the softmax weights are
//     rounded to the element type before those products,
//   - exp() is the ex2.approx expansion, and
//   - history * gain is the initial accumulator of the value product rather
//     than an addend applied afterwards.
// Every reduction has a fixed order and there are no atomics, so the kernel
// is deterministic.
//
// The fragments are loaded with ldmatrix and fed to mma.sync directly: with
// this toolchain the wmma bf16 API lowers its fragment loads to per-element
// generic loads, and owning the layouts keeps a row's softmax in registers.
//
// One thread block serves kCols value columns of one (request, value head)
// with two warps per 16-column tile, a (row half, column tile) grid over the
// 32 rows x kCols columns. A block is bound by the latency of its dependent
// instructions and of memory, so it is laid out as a short critical path:
//   1. issue every load before anything waits on one;
//   2. stage normalized q, k and the values, then one block barrier;
//   3. the first column tile of each row half computes all 32 scores of its
//      16 rows and their softmax, while the state tile is still in flight;
//   4. stage the rounded state tile, one more block barrier;
//   5. every warp, without further barriers: q . state for its tile, then
//      W . values accumulated on top of the scaled history, and the
//      write-back.
// A lane holds two rows of every 16-row tile, so a row's softmax is a
// reduction over four lanes. A wide column tile keeps the work per head
// from being repeated, which is what a full GPU is bound by; a narrow one
// spreads a single request over more SMs.
//
// Grid: x = value_dim / kCols, y = requests, z = value heads; dynamic shared
// memory: Tiling::kSmemBytes.
// The layout requirements (contiguous 16-byte aligned rows, indexes and
// strides below 2^31, power-of-two head sharing, element-typed scalars) are
// checked by the host; anything else takes the shared SIMT kernel.

constexpr unsigned kFull = 0xffffffffu;
constexpr int kDim = 128;    // KEY_DIM == VALUE_DIM == 128 (validated)
constexpr int kRows = 32;    // block_size <= 32 (validated)
constexpr int kMmaM = 16;    // mma.m16n8k16
constexpr int kMmaN = 8;
constexpr int kPad = 8;      // 16-byte row pad; ldmatrix stays conflict free
constexpr int kLd = kDim + kPad;
constexpr int kLdW = kRows + kPad;  // softmax weights: [row][key]
constexpr int kSteps = kDim / kMmaM;      // K-steps of q . state / q . k
constexpr int kKeyTiles = kRows / kMmaM;  // 16-key tiles of the scores
constexpr int kValueSteps = kKeyTiles;    // K-steps of W . values

template <int kCols, int kThreads>
struct Tiling {
  static_assert(kCols % kMmaM == 0 && kCols <= kDim, "kCols must tile 128");
  static constexpr int kColTiles = kCols / kMmaM;
  static constexpr int kWarps = kThreads / 32;
  static_assert(kWarps == 2 * kColTiles, "two warps per 16-column tile");
  static constexpr int kLdV = kCols + kPad;  // values: [key row][column]
  // 16-byte units per row must be odd for conflict-free ldmatrix.
  static_assert((kLdV / 8) % 2 == 1, "odd number of 16-byte units per row");
  static_assert(kThreads % kRows == 0, "whole row groups");
  // A row group's threads are consecutive lanes of one warp.
  static constexpr int kThreadsPerRow = kThreads / kRows;
  static_assert(kThreadsPerRow > 0 && kThreadsPerRow <= 32 &&
                    (kThreadsPerRow & (kThreadsPerRow - 1)) == 0,
                "a row's threads are consecutive lanes of one warp");
  static constexpr int kChunks = kDim / 8 / kThreadsPerRow;  // 8 elements each
  static_assert(kChunks >= 1, "at most one thread per eight elements");
  static constexpr int kOut = kCols / kThreadsPerRow;  // tile columns per thread
  static_assert(kOut == 8, "each thread stages one 16-byte chunk of values");
  // State staging: a thread stages kStateF4 float4 (32 keys of one column).
  static constexpr int kStateF4 = kDim * kCols / kThreads / 4;
  static_assert(kStateF4 >= 2 && kStateF4 % 2 == 0 && kStateF4 * 4 * kThreads == kDim * kCols,
                "each thread stages a whole number of 8-key chunks");
  // Column pairs of the block, four per thread; an inactive request zeroes
  // them through the same mapping as the write-back uses rows.
  static constexpr int kZeroWrites = kRows * (kCols / 2) / kThreads;
  static_assert(kZeroWrites * kThreads == kRows * (kCols / 2),
                "each thread zeroes a whole number of column pairs");
  // Shared memory layout, offsets in elements; every segment starts on a
  // 16-byte boundary. The last segment holds the valid flags, the history
  // logit and the per-row history scale.
  static constexpr int kKOffset = kRows * kLd;
  static constexpr int kSOffset = kKOffset + kRows * kLd;
  static constexpr int kVOffset = kSOffset + kCols * kLd;
  static constexpr int kWOffset = kVOffset + kRows * kLdV;
  static constexpr int kScalarsOffset = kWOffset + kRows * kLdW;
  static constexpr int kSmemBytes = kScalarsOffset * 2 + 192;
  static_assert(kSmemBytes % 16 == 0, "aligned shared memory size");
  // The write-back collects each warp's tile in the k tile.
  static_assert(kWarps * kMmaM * kMmaM <= kRows * kLd,
                "a staging slot per warp fits in the k tile");
};

struct __align__(16) Words4 {
  uint32_t w[4];
};

// Read-only global loads: one instruction for a 16-byte group of elements.
__device__ __forceinline__ Words4 LoadWords4(const void* src) {
  const uint4 v = __ldg(reinterpret_cast<const uint4*>(src));
  return Words4{v.x, v.y, v.z, v.w};
}
__device__ __forceinline__ float4 LoadFloat4(const float* src) {
  return __ldg(reinterpret_cast<const float4*>(src));
}

// One packed pair of elements, in both directions.
template <typename Data>
__device__ __forceinline__ float2 ToFloat2(uint32_t pair);
template <>
__device__ __forceinline__ float2 ToFloat2<__nv_bfloat16>(uint32_t pair) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&pair));
}
template <>
__device__ __forceinline__ float2 ToFloat2<half>(uint32_t pair) {
  return __half22float2(*reinterpret_cast<const __half2*>(&pair));
}
template <typename Data>
__device__ __forceinline__ uint32_t FromFloat2(float x, float y);
template <>
__device__ __forceinline__ uint32_t FromFloat2<__nv_bfloat16>(float x, float y) {
  const __nv_bfloat162 pair = __floats2bfloat162_rn(x, y);
  return *reinterpret_cast<const uint32_t*>(&pair);
}
template <>
__device__ __forceinline__ uint32_t FromFloat2<half>(float x, float y) {
  const __half2 pair = __floats2half2_rn(x, y);
  return *reinterpret_cast<const uint32_t*>(&pair);
}

// Single elements, in both directions.
template <typename Data>
struct Scalar;
template <>
struct Scalar<__nv_bfloat16> {
  static __device__ __forceinline__ float To(__nv_bfloat16 value) {
    return __bfloat162float(value);
  }
  static __device__ __forceinline__ __nv_bfloat16 From(float value) {
    return __float2bfloat16_rn(value);
  }
};
template <>
struct Scalar<half> {
  static __device__ __forceinline__ float To(half value) {
    return __half2float(value);
  }
  static __device__ __forceinline__ half From(float value) {
    return __float2half_rn(value);
  }
};

// The reference lowers exp() to the ex2.approx.f32 hardware approximation;
// exp2f() is that same instruction.
__device__ __forceinline__ float FastExp(float x) {
  return exp2f(x * 1.4426950408889634f);
}

// True when every value head shares its q/k head with a power of two of
// value heads, which the tensor-core kernel needs: the mapping is then a
// shift instead of an integer division at the head of every block.
inline bool PowerOfTwoRatio(int num_v_heads, int num_heads) {
  const int ratio = num_v_heads / num_heads;
  return num_heads > 0 && num_v_heads % num_heads == 0 &&
         (ratio & (ratio - 1)) == 0;
}
inline int RatioShift(int num_v_heads, int num_heads) {
  int ratio = num_v_heads / num_heads, shift = 0;
  while (ratio > 1) {
    ratio >>= 1;
    ++shift;
  }
  return shift;
}

// index * stride in elements. The host has checked that indexes and strides
// of the call fit in 32 bits (and routes wider layouts to the shared SIMT
// kernel), so the product is one wide multiply instead of a 64-bit one at
// the head of every address.
__device__ __forceinline__ int64_t Offset(int64_t index, int64_t stride) {
  return static_cast<int64_t>(static_cast<uint64_t>(static_cast<uint32_t>(index)) *
                              static_cast<uint32_t>(stride));
}

// Normalizes one staged row and writes it to shared memory: the
// threads_per_row consecutive lanes of a warp hold chunks x 8 elements each,
// reduce the square sum in a fixed order and round the scaled elements like
// the reference.
template <typename Data, int kThreadsPerRow, int kChunks>
__device__ __forceinline__ void NormStoreRow(const Words4 (&raw)[kChunks],
                                             Data* __restrict__ dst,
                                             int part) {
  float2 f[kChunks][4];
  float sumsq = 0.0f;
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      f[c][p] = ToFloat2<Data>(raw[c].w[p]);
      sumsq += f[c][p].x * f[c][p].x + f[c][p].y * f[c][p].y;
    }
  }
#pragma unroll
  for (int off = 1; off < kThreadsPerRow; off <<= 1)
    sumsq += __shfl_xor_sync(kFull, sumsq, off);
  const float norm = rsqrtf(sumsq + 1.0e-6f);
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    Words4 out;
#pragma unroll
    for (int p = 0; p < 4; ++p)
      out.w[p] = FromFloat2<Data>(f[c][p].x * norm, f[c][p].y * norm);
    *reinterpret_cast<Words4*>(dst + 8 * (c * kThreadsPerRow + part)) = out;
  }
}

// ---- warp-level mma.m16n8k16 with fp32 accumulation ----
// Lane l (group g = l >> 2, tig = l & 3) holds:
//   A (16x16, row-major):   a0 = A[g][2 tig .. +1], a1 = A[g + 8][2 tig ..],
//                            a2, a3 = the same eight columns further
//   B (16x8, column-major): b0 = B[2 tig .. +1][g], b1 = eight rows further
//   D (16x8):               d0, d1 = D[g][2 tig .. +1], d2, d3 = D[g + 8][..]

__device__ __forceinline__ uint32_t SharedAddress(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Four 8x8 matrices; lane l names a 16-byte row of matrix l / 8.
__device__ __forceinline__ void LoadMatrix4(uint32_t (&r)[4], uint32_t address) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(address)
      : "memory");
}
// The same, each 8x8 matrix transposed on the way to the registers.
__device__ __forceinline__ void LoadMatrix4Trans(uint32_t (&r)[4],
                                                 uint32_t address) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(address)
      : "memory");
}

// A of one 16x16 tile from a [row][kLd] array; `tile` is its first element.
template <typename Data, int kLd>
__device__ __forceinline__ void LoadAFrag(uint32_t (&a)[4], const Data* tile) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  LoadMatrix4(a, SharedAddress(tile +
                               ((lane & 7) + 8 * ((lane >> 3) & 1)) * kLd +
                               8 * (lane >> 4)));
}

// Four B tiles of a 16 x 16 block, K rows 0..7 and 8..15 by N columns 0..7
// and 8..15; b[0], b[1] serve the first 8 columns, b[2], b[3] the next 8.
// A lane ends up with the two consecutive K elements of its column.
//   LoadBFrags:      B[k][n] is stored as tile[n][k] (K contiguous).
//   LoadBFragsTrans: B[k][n] is stored as tile[k][n] (N contiguous).
template <typename Data, int kLd>
__device__ __forceinline__ void LoadBFrags(uint32_t (&b)[4],
                                           const Data* tile) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  LoadMatrix4(b, SharedAddress(tile +
                               ((lane & 7) + 8 * (lane >> 4)) * kLd +
                               8 * ((lane >> 3) & 1)));
}
template <typename Data, int kLd>
__device__ __forceinline__ void LoadBFragsTrans(uint32_t (&b)[4],
                                                const Data* tile) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  LoadMatrix4Trans(b, SharedAddress(tile +
                                    ((lane & 7) + 8 * ((lane >> 3) & 1)) * kLd +
                                    8 * (lane >> 4)));
}

// d = a . b + d
template <typename Data>
__device__ __forceinline__ void Mma16816(float (&d)[4], const uint32_t (&a)[4],
                                         uint32_t b0, uint32_t b1) {
  if constexpr (std::is_same_v<Data, __nv_bfloat16>) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  } else {
    static_assert(std::is_same_v<Data, half>, "bf16 or f16");
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
}

template <typename Data, typename Index, int kCols, int kThreads,
          int kMinBlocks>
__global__ void __launch_bounds__(kThreads, kMinBlocks)
    CloverLocalMemoryMmaKernel(
        const Data* __restrict__ q, const Data* __restrict__ k,
        const Data* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ state, const Index* __restrict__ indices,
        const int8_t* __restrict__ valid, const Data* __restrict__ logit_scale,
        const Data* __restrict__ history_bias,
        const Data* __restrict__ history_c, Data* __restrict__ out,
        int64_t q_stride_t, int64_t q_stride_h, int64_t k_stride_t,
        int64_t k_stride_h, int64_t v_stride_t, int64_t v_stride_h,
        int64_t g_stride_t, int64_t g_stride_h, int64_t state_slot_stride,
        int64_t state_h_stride, int64_t state_v_stride, int64_t out_t_stride,
        int64_t out_h_stride, int64_t out_v_stride, float scale,
        float reference_block_size, int block_size, int num_slots,
        int q_head_shift, int k_head_shift) {
  using T = Tiling<kCols, kThreads>;
  constexpr int kTpr = T::kThreadsPerRow;
  constexpr int kPairs = kCols / 2;  // adjacent column pairs per row
  const int c0 = static_cast<int>(blockIdx.x) * kCols;
  const int request = static_cast<int>(blockIdx.y);
  const int value_head = static_cast<int>(blockIdx.z);
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t row_base = static_cast<int64_t>(request) * block_size;
  // This thread's warp in the (row half, column tile) grid of the products,
  // and its lane's two rows of every 16-row tile.
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int row_half = warp % 2;  // rows [16 * row_half, +16)
  const int col_tile = warp / 2;  // columns [16 * col_tile, +16)
  const int group = lane >> 2;
  const int tig = lane & 3;
  // The first column tile of a row half also owns the scores and the softmax
  // of the row half.
  const bool scorer = col_tile == 0;
  const int r0 = 16 * row_half + group;  // the lane's rows: r0 and r0 + 8
  // One row group per kTpr threads: the q/k/value row it stages is also the
  // row whose gate, softmax and output it handles.
  const int row = tid / kTpr;
  const int part = tid % kTpr;
  const bool row_live = row < block_size;

  // ---- issue every load before anything waits on one ----
  // Nothing below may consume a loaded value: in-order issue would stall the
  // warp for the full memory latency in the middle of the address
  // arithmetic. The per-head scalars need nothing but the head and stay in
  // their element type until they are used; the slot index leads because the
  // state tile depends on it.
  const Data ls_raw = __ldg(logit_scale + value_head);
  const Data history_bias_raw = __ldg(history_bias + value_head);
  const Data history_c_raw = __ldg(history_c + value_head);
  const Index cache_index = __ldg(indices + request);
  // The gates of the lane's two rows.
  const float* g_src = g + Offset(row_base + r0, g_stride_t) +
                       Offset(value_head, g_stride_h);
  const float gate_row0 = __ldg(g_src);
  const float gate_row1 = __ldg(g_src + Offset(8, g_stride_t));
  // The thread's own valid flag (thread j owns key j).
  int8_t own_valid = 0;
  if (tid < block_size) own_valid = __ldg(valid + row_base + tid);

  // Value heads may share q/k heads; the host passes the shift of the
  // sharing ratio.
  const int q_head = value_head >> q_head_shift;
  const int k_head = value_head >> k_head_shift;
  Words4 q_raw[T::kChunks], k_raw[T::kChunks], v_raw;
  {
    const Data* q_src =
        q + Offset(row_base + row, q_stride_t) +
            Offset(q_head, q_stride_h) + 8 * part;
    const Data* k_src =
        k + Offset(row_base + row, k_stride_t) +
            Offset(k_head, k_stride_h) + 8 * part;
    const Data* v_src =
        v + Offset(row_base + row, v_stride_t) +
            Offset(value_head, v_stride_h) + c0 + 8 * part;
    const Words4 zero{0u, 0u, 0u, 0u};
#pragma unroll
    for (int c = 0; c < T::kChunks; ++c) {
      q_raw[c] = row_live ? LoadWords4(q_src + 8 * c * kTpr) : zero;
      k_raw[c] = row_live ? LoadWords4(k_src + 8 * c * kTpr) : zero;
    }
    v_raw = row_live ? LoadWords4(v_src) : zero;
  }

  // An inactive request reads slot 0 and writes zeros below: a return that
  // waits for a load would let the compiler sink every load after it.
  const bool request_active =
      cache_index >= 0 && cache_index < static_cast<Index>(num_slots);
  const int64_t slot = request_active ? static_cast<int64_t>(cache_index) : 0;
  // The output base is wanted by the zero fill and the write-back only.
  auto head_base = [&]() {
    return out + Offset(row_base, out_t_stride) +
           Offset(value_head, out_h_stride);
  };

  // Without any slot the reference output is all zeros. The test depends on
  // a kernel argument alone, so it cannot delay the loads above either.
  if (num_slots <= 0) {
    Data* const head_out = head_base();
#pragma unroll
    for (int i = 0; i < T::kZeroWrites; ++i) {
      const int item = tid + i * kThreads;
      const int r = item / kPairs;
      if (r < block_size) {
        Data* const dst = head_out + Offset(r, out_t_stride) +
                          Offset(c0 + 2 * (item % kPairs), out_v_stride);
        dst[0] = Scalar<Data>::From(0.0f);
        dst[out_v_stride] = Scalar<Data>::From(0.0f);
      }
    }
    return;
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  extern __shared__ __align__(32) unsigned char smem[];
  Data* const q_s = reinterpret_cast<Data*>(smem);
  Data* const k_s = reinterpret_cast<Data*>(smem + T::kKOffset * sizeof(Data));
  Data* const s_s = reinterpret_cast<Data*>(smem + T::kSOffset * sizeof(Data));
  Data* const v_s = reinterpret_cast<Data*>(smem + T::kVOffset * sizeof(Data));
  Data* const w_s = reinterpret_cast<Data*>(smem + T::kWOffset * sizeof(Data));
  unsigned char* const scalars = smem + T::kScalarsOffset * sizeof(Data);
  int8_t* const valid_s = reinterpret_cast<int8_t*>(scalars);
  float* const history_logit_s = reinterpret_cast<float*>(scalars + 32);
  float* const history_scale_s = reinterpret_cast<float*>(scalars + 64);

  // State: one column slice per thread, whole 8-key chunks (kStateF4
  // float4), staged as soon as the first barrier releases.
  const int column = tid % kCols;
  const int slice = tid / kCols;
  float4 s_raw[T::kStateF4];
  {
    const float* s_src =
        state + Offset(slot, state_slot_stride) +
        Offset(value_head, state_h_stride) +
        Offset(c0 + column, state_v_stride) + 4 * T::kStateF4 * slice;
#pragma unroll
    for (int c = 0; c < T::kStateF4; ++c) s_raw[c] = LoadFloat4(s_src + 4 * c);
  }
  if (tid < kRows) valid_s[tid] = own_valid;  // zero past the block

  // ---- stage normalized q and k rows and the values ----
  NormStoreRow<Data, kTpr, T::kChunks>(q_raw, q_s + row * kLd, part);
  NormStoreRow<Data, kTpr, T::kChunks>(k_raw, k_s + row * kLd, part);
  *reinterpret_cast<Words4*>(v_s + row * T::kLdV + 8 * part) = v_raw;
  // The history bucket is sized against the valid block size. The flags are
  // warp 0's lanes, so that warp has the count without a barrier and the
  // logarithm runs beside the other warps' staging.
  if (warp == 0) {
    const int valid_count = __popc(__ballot_sync(kFull, own_valid != 0));
    *history_logit_s =
        Scalar<Data>::To(history_bias_raw) +
        Scalar<Data>::To(history_c_raw) *
            logf(fmaxf(static_cast<float>(valid_count), 1.0f) /
                 reference_block_size);
  }
  // q, k, values and the flags are staged.
  __syncthreads();
  const float history_logit = *history_logit_s;
  // ---- stage the state tile, rounded to the element type ----
  // It has been in flight since the slot index arrived, about as long as
  // the staging above took, and releasing its registers here keeps the
  // scores and the softmax lean.
  {
    Data* dst = s_s + column * kLd + 8 * (T::kStateF4 / 2) * slice;
#pragma unroll
    for (int c = 0; c < T::kStateF4 / 2; ++c) {
      Words4 out;
      out.w[0] = FromFloat2<Data>(s_raw[2 * c].x, s_raw[2 * c].y);
      out.w[1] = FromFloat2<Data>(s_raw[2 * c].z, s_raw[2 * c].w);
      out.w[2] = FromFloat2<Data>(s_raw[2 * c + 1].x, s_raw[2 * c + 1].y);
      out.w[3] = FromFloat2<Data>(s_raw[2 * c + 1].z, s_raw[2 * c + 1].w);
      *reinterpret_cast<Words4*>(dst + 8 * c) = out;
    }
  }

  // ---- scores and softmax of the row half, once ----
  const Data* q_tile = q_s + 16 * row_half * kLd;
  if (scorer) {
    const float ls = Scalar<Data>::To(ls_raw);
    // Both key tiles share the q fragments, which stream one K-step at a
    // time: holding all eight would cost 32 registers.
    float score[kKeyTiles][2][4] = {};
    const Data* k_tile0 = k_s;
    const Data* k_tile1 = k_s + 16 * kLd;
#pragma unroll
    for (int step = 0; step < kSteps; ++step) {
      uint32_t a[4];
      uint32_t b0[4];
      uint32_t b1[4];
      LoadAFrag<Data, kLd>(a, q_tile + kMmaM * step);
      LoadBFrags<Data, kLd>(b0, k_tile0 + kMmaM * step);
      LoadBFrags<Data, kLd>(b1, k_tile1 + kMmaM * step);
      Mma16816<Data>(score[0][0], a, b0[0], b0[1]);
      Mma16816<Data>(score[0][1], a, b0[2], b0[3]);
      Mma16816<Data>(score[1][0], a, b1[0], b1[1]);
      Mma16816<Data>(score[1][1], a, b1[2], b1[3]);
    }
    // A row's 32 logits are spread over its four lanes: key 16 t + 8 n +
    // 2 tig (+1). Dead keys and keys past the block carry -inf. The lane's
    // two rows are independent chains and are interleaved.
    const float gate_rows[2] = {scale * FastExp(gate_row0),
                                scale * FastExp(gate_row1)};
    float logit[2][kKeyTiles][2][2];  // [row][t][n][e]
    float row_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int t = 0; t < kKeyTiles; ++t) {
#pragma unroll
      for (int n = 0; n < 2; ++n) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int key = kMmaM * t + kMmaN * n + 2 * tig + e;
          const bool live = key < block_size && valid_s[key] != 0;
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const float value = score[t][n][2 * half + e];
            logit[half][t][n][e] = live ? value * ls : -INFINITY;
            row_max[half] = fmaxf(row_max[half], logit[half][t][n][e]);
          }
        }
      }
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      // Over the row's four lanes, in a fixed order.
      row_max[half] = fmaxf(row_max[half], __shfl_xor_sync(kFull, row_max[half], 1));
      row_max[half] = fmaxf(row_max[half], __shfl_xor_sync(kFull, row_max[half], 2));
      row_max[half] = fmaxf(row_max[half], history_logit);
    }
    float local_sum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int t = 0; t < kKeyTiles; ++t) {
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            logit[half][t][n][e] = FastExp(logit[half][t][n][e] - row_max[half]);
            local_sum[half] += logit[half][t][n][e];
          }
        }
      }
      local_sum[half] += __shfl_xor_sync(kFull, local_sum[half], 1);
      local_sum[half] += __shfl_xor_sync(kFull, local_sum[half], 2);
    }
    // out = history * (scale * exp(g) * history_weight) + W . values, where
    // the history term is the initial accumulator of the value product.
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const float exp_history = FastExp(history_logit - row_max[half]);
      const float denominator = local_sum[half] + exp_history;
      // The reference divides with the approximate div.full.f32 too.
      const float inverse = __fdividef(1.0f, denominator);
      // The reference rounds the local weights to the element type.
      Data* w_row = w_s + (r0 + 8 * half) * kLdW;
#pragma unroll
      for (int t = 0; t < kKeyTiles; ++t) {
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          *reinterpret_cast<uint32_t*>(w_row + kMmaM * t + kMmaN * n + 2 * tig) =
              FromFloat2<Data>(logit[half][t][n][0] * inverse,
                               logit[half][t][n][1] * inverse);
        }
      }
      history_scale_s[r0 + 8 * half] = (exp_history * inverse) * gate_rows[half];
    }
  }

  // The weights of every row are staged.
  __syncthreads();

  // ---- history = q . state: B[key][column] = state[column][key] ----
  // The q fragments stream one K-step at a time, as above.
  float result[2][4] = {};
  {
    const Data* s_tile = s_s + 16 * col_tile * kLd;
#pragma unroll
    for (int step = 0; step < kSteps; ++step) {
      uint32_t a[4];
      uint32_t b[4];
      LoadAFrag<Data, kLd>(a, q_tile + kMmaM * step);
      LoadBFrags<Data, kLd>(b, s_tile + kMmaM * step);
      Mma16816<Data>(result[0], a, b[0], b[1]);
      Mma16816<Data>(result[1], a, b[2], b[3]);
    }
  }
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    result[n][0] *= history_scale_s[r0];
    result[n][1] *= history_scale_s[r0];
    result[n][2] *= history_scale_s[r0 + 8];
    result[n][3] *= history_scale_s[r0 + 8];
  }
  // The weight fragment reads below need this warp's stores only.
  __syncwarp();

  // ---- values, accumulated on top of the scaled history ----
  {
    const Data* w_tile = w_s + 16 * row_half * kLdW;
    const Data* v_tile = v_s + 16 * col_tile;
#pragma unroll
    for (int step = 0; step < kValueSteps; ++step) {
      uint32_t wa[4];
      uint32_t b[4];
      LoadAFrag<Data, kLdW>(wa, w_tile + kMmaM * step);
      // K is the row axis of the value tile: a step moves 16 rows.
      LoadBFragsTrans<Data, T::kLdV>(b, v_tile + kMmaM * step * T::kLdV);
      Mma16816<Data>(result[0], wa, b[0], b[1]);
      Mma16816<Data>(result[1], wa, b[2], b[3]);
    }
  }

  // ---- write-back ----
  {
    // The reference zeroes inactive requests and the rows that are not
    // active in this step; rows past the block are not written at all.
    const bool keep[2] = {request_active && valid_s[r0] != 0,
                          request_active && valid_s[r0 + 8] != 0};
    Data* const tile0 = head_base() + Offset(16 * row_half, out_t_stride) +
                        Offset(c0 + 16 * col_tile, out_v_stride);
    const bool adjacent = out_v_stride == 1;
    if (adjacent && (out_t_stride & 7) == 0 &&
        (reinterpret_cast<uintptr_t>(tile0) & 15) == 0) {
      // Whole 16-byte stores: the warp collects its 16 x 16 tile in the k
      // staging area, then writes each row's 32 bytes with two lanes.
      // Scattered 4-byte stores cost the memory system several times their
      // data once the GPU is busy. The k tile is dead: its last readers,
      // the scores, finished before the second barrier, unlike q which the
      // other warps of a row half may still be reading for their history.
      uint32_t* const stage = reinterpret_cast<uint32_t*>(k_s) + warp * (kMmaM * kMmaM / 2);
#pragma unroll
      for (int n = 0; n < 2; ++n) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          stage[(group + 8 * half) * (kMmaM / 2) + 4 * n + tig] = FromFloat2<Data>(
              keep[half] ? result[n][2 * half] : 0.0f,
              keep[half] ? result[n][2 * half + 1] : 0.0f);
        }
      }
      __syncwarp();
      const Words4 words = *reinterpret_cast<const Words4*>(stage + 4 * lane);
      if (16 * row_half + (lane >> 1) < block_size)
        *reinterpret_cast<Words4*>(tile0 + Offset(lane >> 1, out_t_stride) + 8 * (lane & 1)) = words;
    } else {
      Data* const row0 = tile0 + Offset(group, out_t_stride) + 2 * tig * out_v_stride;
      const bool pack = adjacent && (out_t_stride & 1) == 0 &&
                        (reinterpret_cast<uintptr_t>(row0) & 3) == 0;
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        Data* const dst0 = row0 + Offset(8 * n, out_v_stride);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          if (r0 + 8 * half >= block_size) continue;
          Data* const dst = dst0 + Offset(8 * half, out_t_stride);
          const float x0 = keep[half] ? result[n][2 * half] : 0.0f;
          const float x1 = keep[half] ? result[n][2 * half + 1] : 0.0f;
          if (pack) {
            *reinterpret_cast<uint32_t*>(dst) = FromFloat2<Data>(x0, x1);
          } else {
            dst[0] = Scalar<Data>::From(x0);
            dst[out_v_stride] = Scalar<Data>::From(x1);
          }
        }
      }
    }
  }
#endif
}

}  // namespace infini::ops::nvidia_clover_local_memory_detail

#endif  // INFINI_OPS_NVIDIA_CLOVER_LOCAL_MEMORY_KERNEL_CUH_
