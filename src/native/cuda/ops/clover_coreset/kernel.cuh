#ifndef INFINI_OPS_CUDA_CLOVER_CORESET_KERNEL_CUH_
#define INFINI_OPS_CUDA_CLOVER_CORESET_KERNEL_CUH_

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "data_type.h"

namespace infini::ops {

// The lane-to-lane shared-memory handoffs below synchronize with
// __syncthreads(), not __syncwarp(): on gfx936 (DTK) __syncwarp does not
// order shared-memory stores between lanes of a warp, so a value lane 0
// publishes for the warp reads back stale in the other lanes. Every barrier
// here stands in code the whole block reaches, so the block-wide barrier is
// also correct, on NVIDIA as well.

// Shared-layer SIMT kernel for CloverCoreset, mirroring the three Triton
// stages of the reference in one launch per routing block:
//   (0) every row's softmax over its 256 experts and its top-8 experts,
//       accumulated into a routing mass per expert;
//   (1) the experts sorted by descending mass (ties by expert id) and kept
//       while the cumulative normalized mass stays below top_p;
//   (2) the allow list applied to the block's logits.
// The reference is three launches that hand a mass and an allow list through
// memory. A block here is at most 32 rows x 256 logits, which one thread
// block keeps in its shared memory: one warp per row, eight warps for the
// sort.
//
// The reference decides with fp32 sums, and a prefix sum that lands on top_p
// is decided by the order those sums associate. So every reduction below
// associates like the reference's generated code (the comment at each one
// spells the order out), which makes the two agree wherever their divisions
// round alike. Nothing is atomic, so the kernel is deterministic.
//
// Lanes mostly hand data over through shared memory rather than warp
// shuffles: on the GPUs this was measured on a warp issues a shuffle only
// every few dozen cycles. A row's top-8 is a merge tree in registers with
// one shared-memory round trip per level, and the sort ranks keys by
// counting and searching instead of passing them through a network of 36
// compare-exchange steps.
//
// The warp barriers all stand in code that every lane of the warp reaches.
// There a compiler that can see the warp is converged needs nothing for
// them; inside a branch that only some lanes take it has to reconverge the
// warp by hand, which costs more than the work between the barriers.
//
// Grid: x = routing blocks.  Block: max(rows, 8) warps of 32 lanes.  Shared
// memory: kSmemBytes.

constexpr int kCloverCoresetExperts = 256;

namespace clover_coreset_detail {

constexpr unsigned kFull = 0xffffffffu;  // the logical warp: 32 aligned lanes
constexpr int kWarp = 32;
constexpr int kPerLane = kCloverCoresetExperts / kWarp;  // experts per lane: 8
constexpr int kTop = 8;                  // experts a row routes to
constexpr int kSortWarps = kCloverCoresetExperts / kWarp;  // one expert per lane
constexpr int kScanWarps = kCloverCoresetExperts / 2 / kWarp;  // two per lane
constexpr int kMaxRows = 32;
constexpr int kRunBits = 5;              // log2(kWarp): a binary search's steps
static_assert((1 << kRunBits) == kWarp, "a run holds one warp of keys");

// Shared memory, in this order: the softmax shares of the rows (each row's
// slice is its warp's scratch until they are written), the allow list, the
// warp sums of the two reductions over the experts, the experts' masses,
// the sorted keys and the eight sorted runs of 32 keys the sort merges.
constexpr int kShareBytes = kMaxRows * kCloverCoresetExperts * sizeof(float);
constexpr int kAllowBytes = kShareBytes;
constexpr int kTotalsBytes = kAllowBytes + kCloverCoresetExperts;
constexpr int kMassBytes =
    kTotalsBytes + 2 * kScanWarps * sizeof(float);
constexpr int kKeysBytes = kMassBytes + kCloverCoresetExperts * sizeof(float);
constexpr int kRunsBytes =
    kKeysBytes + kCloverCoresetExperts * sizeof(unsigned long long);
constexpr int kSmemBytes =
    kRunsBytes + kCloverCoresetExperts * sizeof(unsigned long long);

// The reference lowers exp() to the ex2.approx.f32 hardware approximation.
__device__ __forceinline__ float FastExp(float x) {
  return exp2f(x * 1.4426950408889634f);
}

// A row's candidates carry their expert index in the comparison. The order
// key of an IEEE float is its sign and magnitude mapped onto a monotone
// unsigned integer; a key keeps the top bits of that order and, in its low
// 8 bits, 255 - expert. The largest key is so the best logit and, at equal
// logits, the lower expert id: the reference's tie break.
//
// The tree below takes 32-bit keys. A 16-bit logit has at least 13 zero
// mantissa bits once it is a float, so its top 19 bits order it exactly and
// its 32-bit key is exact. An fp32 logit does not fit: the kernel runs the
// tree on a key that drops the low 8 mantissa bits and checks afterwards
// whether that could have mattered (the kernel body says how); when it
// could, the rounds decide, on the exact 64-bit key.
constexpr int kExpertBits = 8;

// The order key: sign and magnitude onto a monotone unsigned integer.
__device__ __forceinline__ uint32_t OrderKey(float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <typename Data>
__device__ __forceinline__ uint32_t MakeKey(float x, int expert) {
  constexpr int kDrop = sizeof(Data) == 2 ? 13 : kExpertBits;
  return ((OrderKey(x) >> kDrop) << kExpertBits) |
         static_cast<unsigned>(255 - expert);
}

__device__ __forceinline__ unsigned long long MakeExactKey(float x,
                                                           int expert) {
  return (static_cast<unsigned long long>(OrderKey(x)) << kExpertBits) |
         static_cast<unsigned>(255 - expert);
}

// The exchanges that remain stay inside the 32-lane slice of the logical
// warp: xor offsets below 32 and an explicit width, which is safe on
// warp-64 devices.

template <typename Key>
__device__ __forceinline__ Key WarpMaxKey(Key x) {
#pragma unroll
  for (int off = kWarp / 2; off > 0; off >>= 1) {
    const Key other = __shfl_xor_sync(kFull, x, off, kWarp);
    x = other > x ? other : x;
  }
  return x;
}

// ---- the top-8 of a row: a merge tree on its 32-bit keys ----
// The unsigned min and max order the keys, one instruction each.

// One step of a bitonic network over the lane's own keys: position i
// exchanges with i ^ kDistance. The blocks of kSize keys alternate
// direction and the block that holds position 0 sorts descending.
template <int kSize, int kDistance>
__device__ __forceinline__ void LocalStep(uint32_t (&key)[kPerLane]) {
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    const int j = i ^ kDistance;
    if (j > i) {
      const uint32_t a = key[i], b = key[j];
      const bool ascending = (i & kSize) != 0;
      key[i] = ascending ? ::min(a, b) : ::max(a, b);
      key[j] = ascending ? ::max(a, b) : ::min(a, b);
    }
  }
}

__device__ __forceinline__ void LocalSort(uint32_t (&key)[kPerLane]) {
  LocalStep<2, 1>(key);
  LocalStep<4, 2>(key);
  LocalStep<4, 1>(key);
  LocalStep<8, 4>(key);
  LocalStep<8, 2>(key);
  LocalStep<8, 1>(key);
}

// The rest of a merge of the lane's eight keys and eight others: the first
// step of that merge pairs the two lists and is done across lanes.
__device__ __forceinline__ void LocalMerge(uint32_t (&key)[kPerLane]) {
  LocalStep<8, 4>(key);
  LocalStep<8, 2>(key);
  LocalStep<8, 1>(key);
}

// A lane's eight keys, two 16-byte accesses. A lane that is off reads
// zeros and stores nothing.
__device__ __forceinline__ void StoreKeys(uint32_t* slot,
                                          const uint32_t (&key)[kPerLane],
                                          bool on) {
  if (!on) return;
  uint4* const out = reinterpret_cast<uint4*>(slot);
  out[0] = make_uint4(key[0], key[1], key[2], key[3]);
  out[1] = make_uint4(key[4], key[5], key[6], key[7]);
}

__device__ __forceinline__ void LoadKeys(uint32_t (&key)[kPerLane],
                                         const uint32_t* slot, bool on) {
  const uint4* const in = reinterpret_cast<const uint4*>(slot);
  const uint4 zero = make_uint4(0, 0, 0, 0);
  const uint4 low = on ? in[0] : zero, high = on ? in[1] : zero;
  key[0] = low.x;
  key[1] = low.y;
  key[2] = low.z;
  key[3] = low.w;
  key[4] = high.x;
  key[5] = high.y;
  key[6] = high.z;
  key[7] = high.w;
}

// The eighth best key of a row. Every lane starts with its eight keys
// sorted; five merge levels then pair the lanes that still hold
// candidates. A live lane merges its list with its partner's, keeps the
// better eight, and after the last level lane 0 holds the best eight of the
// row. `scratch` is the warp's own 256 words, `key` is sorted on entry.
//
// Every lane runs every level on its own keys, the lanes without candidates
// on zeros: keys that are only updated under a branch some lanes skip end
// up in local memory.
__device__ __forceinline__ uint32_t EighthKey(uint32_t (&key)[kPerLane],
                                              uint32_t* scratch, int lane) {
  uint32_t* const slot = scratch + kPerLane * lane;
  StoreKeys(slot, key, true);
  __syncthreads();
#pragma unroll
  for (int distance = 1; distance < kWarp; distance <<= 1) {
    // Only every distance-th lane still holds candidates.
    const bool live = (lane & (distance - 1)) == 0;
    uint32_t other[kPerLane];
    LoadKeys(other, slot + kPerLane * ((lane ^ distance) - lane), live);
    __syncthreads();
    // The partner's list reversed against this one is a bitonic sequence
    // of sixteen; its first merge step pairs the two lists. Zeros leave
    // a list that is already in order as it is.
#pragma unroll
    for (int r = 0; r < kPerLane; ++r)
      key[r] = ::max(key[r], other[kPerLane - 1 - r]);
    LocalMerge(key);
    StoreKeys(slot, key, live && 2 * distance < kWarp);
    __syncthreads();
  }
  if (lane == 0) slot[0] = key[kPerLane - 1];
  __syncthreads();
  const uint32_t eighth = scratch[0];
  __syncthreads();  // every lane has it: the scratch is free
  return eighth;
}

// The reference's eight rounds of argmax on the exact keys: every round
// takes the best remaining expert of the row. Zero is below the key of any
// logit and is where each maximum starts.
__device__ __forceinline__ uint32_t TopEightByRounds(
    const unsigned long long (&key)[kPerLane], int lane) {
  uint32_t taken = 0;
#pragma unroll
  for (int round = 0; round < kTop; ++round) {
    unsigned long long best = 0;
#pragma unroll
    for (int r = 0; r < kPerLane; ++r)
      if (!((taken >> r) & 1u)) best = key[r] > best ? key[r] : best;
    best = WarpMaxKey(best);
    const int local =
        255 - static_cast<int>(best & 255) - kPerLane * lane;
    if (static_cast<unsigned>(local) < static_cast<unsigned>(kPerLane))
      taken |= 1u << local;
  }
  return taken;
}

// ---- the two reductions of a row's softmax ----
// One lane reduces the 32 values the warp has just written, which is one
// round trip instead of a butterfly of five shuffles. The maximum does not
// depend on the order. The sum is the reference's butterfly (descending
// offsets) as lane 0 sees it, the same tree and so bitwise the same value.
// The trees are spelled out: left to a compiler, the loops over the levels
// stay loops and index registers by a variable.

__device__ __forceinline__ float4 Add4(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __forceinline__ float4 Max4(float4 a, float4 b) {
  return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z),
                     fmaxf(a.w, b.w));
}

__device__ __forceinline__ float RowMaximum(const float* values) {
  const float4* const part = reinterpret_cast<const float4*>(values);
  const float4 low = Max4(Max4(part[0], part[1]), Max4(part[2], part[3]));
  const float4 high = Max4(Max4(part[4], part[5]), Max4(part[6], part[7]));
  const float4 both = Max4(low, high);
  return fmaxf(fmaxf(both.x, both.y), fmaxf(both.z, both.w));
}

// tree[i] takes tree[i + 16], then [i + 8], [i + 4], [i + 2], [i + 1].
__device__ __forceinline__ float RowSum(const float* values) {
  const float4* const part = reinterpret_cast<const float4*>(values);
  const float4 p0 = part[0], p1 = part[1], p2 = part[2], p3 = part[3];
  const float4 p4 = part[4], p5 = part[5], p6 = part[6], p7 = part[7];
  const float4 t16_0 = Add4(p0, p4), t16_1 = Add4(p1, p5);
  const float4 t16_2 = Add4(p2, p6), t16_3 = Add4(p3, p7);
  const float4 t8_0 = Add4(t16_0, t16_2), t8_1 = Add4(t16_1, t16_3);
  const float4 t4 = Add4(t8_0, t8_1);
  const float2 t2 = make_float2(t4.x + t4.z, t4.y + t4.w);
  return t2.x + t2.y;
}

// Eight consecutive elements: 16-byte accesses when the rows allow them
// (else one access per element). A 16-bit element also comes back as its bit pattern (`raw`), which the
// kernel's last step writes out again; its value ranks -0.0 with +0.0, as
// the reference's argmax does.
template <typename Data>
__device__ __forceinline__ void LoadRow(const Data* __restrict__ src,
                                        float (&x)[kPerLane],
                                        uint32_t (&raw)[kPerLane / 2],
                                        bool vector) {
  if constexpr (sizeof(Data) == 2) {
    if (vector) {
      const uint4 bits = __ldg(reinterpret_cast<const uint4*>(src));
      raw[0] = bits.x;
      raw[1] = bits.y;
      raw[2] = bits.z;
      raw[3] = bits.w;
    } else {
#pragma unroll
      for (int r = 0; r < kPerLane; r += 2) {
        const Data pair[2] = {__ldg(src + r), __ldg(src + r + 1)};
        uint16_t low, high;
        std::memcpy(&low, &pair[0], sizeof(low));
        std::memcpy(&high, &pair[1], sizeof(high));
        raw[r / 2] = static_cast<uint32_t>(low) |
                     (static_cast<uint32_t>(high) << 16);
      }
    }
#pragma unroll
    for (int r = 0; r < kPerLane; ++r) {
      uint16_t bits =
          static_cast<uint16_t>(raw[r / 2] >> (16 * (r % 2)));
      Data value;
      std::memcpy(&value, &bits, sizeof(value));
      x[r] = static_cast<float>(value) + 0.0f;
    }
  } else if (vector) {
    const float4* const row = reinterpret_cast<const float4*>(src);
    const float4 low = __ldg(row), high = __ldg(row + 1);
    x[0] = low.x;
    x[1] = low.y;
    x[2] = low.z;
    x[3] = low.w;
    x[4] = high.x;
    x[5] = high.y;
    x[6] = high.z;
    x[7] = high.w;
  } else {
#pragma unroll
    for (int r = 0; r < kPerLane; ++r) x[r] = static_cast<float>(__ldg(src + r));
  }
}

}  // namespace clover_coreset_detail

template <Device::Type kDevice, typename Data>
__global__ void __launch_bounds__(clover_coreset_detail::kMaxRows *
                                    clover_coreset_detail::kWarp)
    CloverCoresetKernel(
        const Data* __restrict__ logits, Data* __restrict__ out, int rows,
        float top_p, int vector) {
  using namespace clover_coreset_detail;
  static_assert(kCloverCoresetExperts == kWarp * kPerLane, "256 experts");

  extern __shared__ __align__(16) unsigned char smem[];
  float* const share = reinterpret_cast<float*>(smem);  // [rows][256]
  uint8_t* const allow = smem + kAllowBytes;
  float* const totals = reinterpret_cast<float*>(smem + kTotalsBytes);
  float* const mass = reinterpret_cast<float*>(smem + kMassBytes);  // [256]
  unsigned long long* const keys =
      reinterpret_cast<unsigned long long*>(smem + kKeysBytes);  // sorted
  unsigned long long* const runs =
      reinterpret_cast<unsigned long long*>(smem + kRunsBytes);  // [8][32]

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & (kWarp - 1);
  const bool in_sort = tid < kCloverCoresetExperts;
  const bool wide = vector != 0;
  const int64_t block_first =
      static_cast<int64_t>(blockIdx.x) * rows * kCloverCoresetExperts;

  // ---- softmax and top-8: one warp per row, lane l owns experts 8l .. ----
  // The row's slice of the shares is the warp's scratch until the end.
  //
  // Every warp of the block walks these steps: one without a row on zero
  // logits and its own, unused, slice. The barriers in between then stand
  // in code the whole warp reaches.
  const bool owns_row = warp < rows;
  float x[kPerLane];
  uint32_t raw[kPerLane / 2];
  if (owns_row) {
    LoadRow<Data>(logits + block_first + warp * kCloverCoresetExperts +
                      kPerLane * lane, x, raw, wide);
  } else {
#pragma unroll
    for (int r = 0; r < kPerLane; ++r) x[r] = 0.0f;
#pragma unroll
    for (int w = 0; w < kPerLane / 2; ++w) raw[w] = 0;
  }
  float local_max = x[0];
#pragma unroll
  for (int r = 1; r < kPerLane; ++r) local_max = fmaxf(local_max, x[r]);
  float* const work = share + warp * kCloverCoresetExperts;

  uint32_t key[kPerLane];
  uint32_t original[kPerLane];
#pragma unroll
  for (int r = 0; r < kPerLane; ++r) {
    // The reference's argmax ranks -0.0 with +0.0; LoadRow has done that
    // for the 16-bit types, whose raw bits the end of the kernel writes.
    const float ordered = sizeof(Data) == 2 ? x[r] : x[r] + 0.0f;
    original[r] = key[r] = MakeKey<Data>(ordered, kPerLane * lane + r);
  }

  work[lane] = local_max;
  __syncthreads();
  LocalSort(key);
  if (lane == 0) work[kWarp] = RowMaximum(work);
  __syncthreads();
  const float row_max = work[kWarp];
  __syncthreads();  // every lane has it: the scratch is free

  // The reference's row reduction: the lane's eight in order, then the
  // butterfly. It normalizes every expert, but only the top-8 masses are
  // summed afterwards, so the division waits for those.
  float weight[kPerLane];
#pragma unroll
  for (int r = 0; r < kPerLane; ++r) weight[r] = FastExp(x[r] - row_max);

  // The eighth best key of the row is the threshold of its top-8.
  uint32_t taken = 0;
  const uint32_t eighth = EighthKey(key, reinterpret_cast<uint32_t*>(work), lane);
#pragma unroll
  for (int r = 0; r < kPerLane; ++r)
    taken |= static_cast<uint32_t>(original[r] >= eighth) << r;
  if constexpr (sizeof(Data) != 2) {
    // The keys dropped the low 8 bits of the order. The eighth key's owner
    // publishes them.
#pragma unroll
    for (int r = 0; r < kPerLane; ++r)
      if (original[r] == eighth)
        reinterpret_cast<int*>(work)[kWarp + 1] =
            static_cast<int>(OrderKey(x[r]) & ((1u << kExpertBits) - 1));
    __syncthreads();
    const int eighth_low = reinterpret_cast<int*>(work)[kWarp + 1];
    __syncthreads();
    // The keys order the row exactly unless a logit outside the top-8 has
    // the same top bits as the eighth one and larger dropped bits.
    int outside = 0;
#pragma unroll
    for (int r = 0; r < kPerLane; ++r)
      outside += ((original[r] >> kExpertBits) == (eighth >> kExpertBits)) &&
                 !((taken >> r) & 1u) &&
                 static_cast<int>(OrderKey(x[r]) & ((1u << kExpertBits) - 1)) >
                     eighth_low;
    reinterpret_cast<int*>(work)[lane] = outside;
    __syncthreads();
    if (lane == 0) {
      const int4* const part = reinterpret_cast<const int4*>(work);
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kWarp / 4; ++i) {
        const int4 four = part[i];
        sum += four.x + four.y + four.z + four.w;
      }
      reinterpret_cast<int*>(work)[kWarp] = sum == 0;
    }
    __syncthreads();
    const bool exact = reinterpret_cast<int*>(work)[kWarp] != 0;
    __syncthreads();
    if (!exact) {
      unsigned long long precise[kPerLane];
#pragma unroll
      for (int r = 0; r < kPerLane; ++r)
        precise[r] = MakeExactKey(x[r] + 0.0f, kPerLane * lane + r);
      taken = TopEightByRounds(precise, lane);
    }
  }

  float partial = weight[0];
#pragma unroll
  for (int r = 1; r < kPerLane; ++r) partial += weight[r];
  work[lane] = partial;
  __syncthreads();
  if (lane == 0) work[kWarp] = RowSum(work);
  __syncthreads();
  const float denominator = work[kWarp];
  __syncthreads();  // every lane has it: the scratch is free

  // Only the experts the row routed to are normalized; a lane has a few.
  float* const own = work + kPerLane * lane;
  reinterpret_cast<float4*>(own)[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  reinterpret_cast<float4*>(own)[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  while (taken) {
    const int r = __ffs(taken) - 1;
    taken &= taken - 1;
    own[r] = weight[r] / denominator;
  }
  __syncthreads();

  // ---- expert masses and the sort's first level: one thread per expert ----
  // The reference reduces over the rows in two steps: its four warps hold
  // the rows w, w + 4, w + 8, .. each and add them in order, and the four
  // partials combine as (P0 + P2) + (P1 + P3).
  unsigned long long sort_key = 0;  // this expert's place in the order
  float expert = 0.0f;         // its mass
  int rank = 0;                // its rank among the warp's 32 experts
  if (in_sort) {
    float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int row = 0; row < kMaxRows; ++row)
      if (row < rows)
        partial[row & 3] += share[row * kCloverCoresetExperts + tid];
    expert = (partial[0] + partial[2]) + (partial[1] + partial[3]);
    sort_key = (static_cast<unsigned long long>(0x7FFFFFFFu -
                                           __float_as_uint(expert))
           << 32) |
          static_cast<unsigned long long>(tid);
    mass[tid] = expert;
  }
  __syncthreads();
  // Each warp orders its own 32 experts by counting the better ones: a
  // larger mass, or the same mass at a lower expert id.
  if (in_sort) {
    const float4* const group =
        reinterpret_cast<const float4*>(mass + warp * kWarp);
#pragma unroll
    for (int i = 0; i < kWarp / 4; ++i) {
      const float4 other = group[i];
      rank += (other.x > expert) || ((other.x == expert) && (4 * i < lane));
      rank += (other.y > expert) || ((other.y == expert) && (4 * i + 1 < lane));
      rank += (other.z > expert) || ((other.z == expert) && (4 * i + 2 < lane));
      rank += (other.w > expert) || ((other.w == expert) && (4 * i + 3 < lane));
    }
    runs[warp * kWarp + rank] = sort_key;
  }
  __syncthreads();

  // ---- block total ----
  // The reference holds two adjacent experts per thread of four warps: it
  // adds the pair, runs a butterfly over the warp's 32 lanes and combines
  // the four warps as (W0 + W2) + (W1 + W3). Threads 0..127 take its
  // layout here, so the butterfly stays inside a warp.
  if (tid < kCloverCoresetExperts / 2) {
    const float2 pair = *reinterpret_cast<const float2*>(mass + 2 * tid);
    float sum = pair.x + pair.y;
#pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1)
      sum += __shfl_xor_sync(kFull, sum, off, kWarp);
    if (lane == 0) totals[warp] = sum;
  }

  // ---- sort by descending mass, ties by expert id ----
  // Eight sorted runs of 32 keys are in place. Every expert counts the keys
  // ahead of it: its own rank and, in each other run, the keys below it,
  // which a binary search finds. That number is its position in the order.
  // The keys are unique, so the positions are too.
  if (in_sort) {
    // The searches step together: their loads are independent.
    int low[kSortWarps], length[kSortWarps];
#pragma unroll
    for (int run = 0; run < kSortWarps; ++run) {
      low[run] = 0;
      length[run] = kWarp;
    }
#pragma unroll
    for (int step = 0; step < kRunBits; ++step) {
#pragma unroll
      for (int run = 0; run < kSortWarps; ++run) {
        if (run != warp) {
          // Halve the interval; its lower end stays inside the run.
          const int half = length[run] >> 1;
          if (runs[run * kWarp + low[run] + half] < sort_key) low[run] += half;
          length[run] -= half;
        }
      }
    }
#pragma unroll
    for (int run = 0; run < kSortWarps; ++run) {
      // One candidate is left: the keys below it and, if it is below too,
      // itself.
      if (run != warp)
        rank += low[run] + (runs[run * kWarp + low[run]] < sort_key);
    }
    keys[rank] = sort_key;
  }
  __syncthreads();

  // ---- the cumulative threshold, in the reference's association ----
  // The reference holds two consecutive sorted experts per thread of four
  // warps (thread t: positions 2t and 2t + 1). It scans the pair sums
  // upwards with the offsets 1, 2, 4, 8, 16 inside each warp, and a warp
  // adds the running total of the warps before it afterwards. The first
  // element's cumulative value adds the exclusive prefix to its own
  // probability. Threads 0..127 take that layout here.
  const float total =
      fmaxf((totals[0] + totals[2]) + (totals[1] + totals[3]), 1.0e-9f);

  unsigned expert0 = 0, expert1 = 0;
  float p0 = 0.0f, p1 = 0.0f, mass0 = 0.0f, mass1 = 0.0f, running = 0.0f;
  const bool in_scan = tid < kCloverCoresetExperts / 2;
  // [0, kScanWarps): the block total's warp sums; then the scan's.
  float* const sums = totals + kScanWarps;
  if (in_scan) {
    const ulonglong2 pair =
        *reinterpret_cast<const ulonglong2*>(keys + 2 * tid);
    expert0 = static_cast<unsigned>(pair.x);
    expert1 = static_cast<unsigned>(pair.y);
    mass0 = mass[expert0];
    mass1 = mass[expert1];
    p0 = mass0 / total;
    p1 = mass1 / total;
    running = p0 + p1;
#pragma unroll
    for (int off = 1; off < kWarp; off <<= 1) {
      const float below = __shfl_up_sync(kFull, running, off, kWarp);
      if (lane >= off) running += below;
    }
    if (lane == kWarp - 1) sums[warp] = running;
  }
  __syncthreads();
  if (in_scan) {
    const float before = warp == 1   ? sums[0]
                         : warp == 2 ? sums[0] + sums[1]
                                     : (sums[0] + sums[1]) + sums[2];
    const float cumulative = warp > 0 ? before + running : running;
    const float previous = __shfl_up_sync(kFull, cumulative, 1, kWarp);
    const float first =
        warp == 0 && lane == 0 ? p0 : p0 + (lane > 0 ? previous : before);
    bool keep0 = first - p0 < top_p;
    bool keep1 = cumulative - p1 < top_p;
    if (top_p >= 1.0f) {  // the reference keeps the experts with mass ...
      keep0 = mass0 > 0.0f;
      keep1 = mass1 > 0.0f;
    }
    if (tid == 0) keep0 = true;  // ... and the first expert of the sort
    allow[expert0] = keep0;
    allow[expert1] = keep1;
  }
  __syncthreads();

  // ---- the allow list applied to the block's logits ----
  // A row's warp still holds the logits it loaded, so nothing is read back.
  if (owns_row) {
    // One flag byte per expert, all eight in one load.
    const uint2 flags =
        *reinterpret_cast<const uint2*>(allow + kPerLane * lane);
    const unsigned long long kept =
        static_cast<unsigned long long>(flags.y) << 32 | flags.x;
    const int64_t first =
        block_first + warp * kCloverCoresetExperts + kPerLane * lane;
    if constexpr (sizeof(Data) == 2) {
      uint16_t infinity;
      const Data negative = static_cast<Data>(-INFINITY);
      std::memcpy(&infinity, &negative, sizeof(infinity));
      uint32_t word[kPerLane / 2];
#pragma unroll
      for (int w = 0; w < kPerLane / 2; ++w) {
        const uint32_t low =
            (kept >> (16 * w)) & 1u ? raw[w] & 0xFFFFu : infinity;
        const uint32_t high =
            (kept >> (16 * w + 8)) & 1u ? raw[w] >> 16 : infinity;
        word[w] = low | (high << 16);
      }
      if (wide) {
        *reinterpret_cast<uint4*>(out + first) =
            make_uint4(word[0], word[1], word[2], word[3]);
      } else {
#pragma unroll
        for (int r = 0; r < kPerLane; ++r) {
          const uint16_t bits =
              static_cast<uint16_t>(word[r / 2] >> (16 * (r % 2)));
          Data value;
          std::memcpy(&value, &bits, sizeof(value));
          out[first + r] = value;
        }
      }
    } else {
      Data value[kPerLane];
#pragma unroll
      for (int r = 0; r < kPerLane; ++r)
        value[r] = ((kept >> (8 * r)) & 0xFFu) != 0 ? x[r] : -INFINITY;
      if (wide) {
        float4* const row = reinterpret_cast<float4*>(out + first);
        row[0] = make_float4(value[0], value[1], value[2], value[3]);
        row[1] = make_float4(value[4], value[5], value[6], value[7]);
      } else {
#pragma unroll
        for (int r = 0; r < kPerLane; ++r) out[first + r] = value[r];
      }
    }
  }
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_CORESET_KERNEL_CUH_
