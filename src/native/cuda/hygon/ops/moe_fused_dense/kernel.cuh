#ifndef INFINI_OPS_HYGON_MOE_FUSED_DENSE_KERNEL_CUH_
#define INFINI_OPS_HYGON_MOE_FUSED_DENSE_KERNEL_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace infini::ops::moe_fused_dense_detail {

// Expert computation of the fused MoE with the matrix instructions of gfx936,
// on the block layout written by MoeAlignBlockSize: sorted[padded row] holds a
// pair index (or the sentinel tokens * topk for padding), experts[block] the
// expert of a block, total[0] the number of padded rows in use.
//
//   W13Kernel    act[pair, f]  = silu(x . w13[e, f, :]) * (x . w13[e, i + f,
//   :]) W2Kernel     pair[pair, h] = weight[pair] * (act[pair] . w2[expert]^T)
//   CombineKernel output[token] = sum over the token's active pairs
//
// One wavefront (64 lanes, one workgroup) streams two 16-row weight bands for
// one expert block: the gate and up rows of 16 features (w13), or two 16-row
// tiles of w2. A k step covers 128 k:
//   - every load instruction reads 256 contiguous bytes from each of 4 rows,
//   which is what the
//     memory system needs to run at its peak bandwidth,
//   - the 8 KB step goes through LDS and comes back in the register layout of
//     v_mmac_f32_16x16x16_{bf16,f16}:
//       A: row = lane % 16, k = (lane / 16) * 4 + 2 * reg + half
//       B: col = lane % 16, k as above          C: row = lane % 16, col = 4 *
//       reg + lane / 16
//     Units of a row are stored XOR-swizzled (unit ^ row) to spread the reads
//     over the banks.
//   - the loads of step s + 1 are in flight while step s is consumed, and at
//   the end of a work
//     item they are the first step of the wavefront's next item, whose metadata
//     was fetched at the start of the current one: the weight stream never
//     drains between items.
// Complete tiles and k steps take a branch-free load path; partial ones at the
// end of a matrix use per-lane predicates.
//
// Requirements: block_size 16, and h and i multiples of 8 (a 16-byte unit holds
// 8 k) of at least 32.

constexpr int kBlock = 16;  // rows of an expert block (block_size)
constexpr int kLanes = 64;  // lanes per wavefront; one wavefront per workgroup
constexpr int kUnit = 8;    // k values per 16-byte unit
constexpr int kStep = 128;  // k values per step: 16 units of a row
constexpr int kBand = 256;  // 16-byte units per LDS band: 16 rows x 16 units
constexpr int kWavesPerCu =
    8;  // wavefronts launched per CU; each owns 8 KB of LDS

using V2I = int __attribute__((ext_vector_type(2)));
using V4F = float __attribute__((ext_vector_type(4)));

// acc += a . b for one 16 x 16 tile over 16 k.
template <typename T>
__device__ __forceinline__ V4F Mma(V2I a, V2I b, V4F c) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
  } else {
    return __builtin_amdgcn_mmac_f32_16x16x16f16(a, b, c);
  }
}

// The 16-bit storage of a float, rounded to nearest even.
template <typename T>
__device__ __forceinline__ uint16_t FromFloat(float f) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(f));
  } else {
    return __half_as_ushort(__float2half_rn(f));
  }
}

template <typename T>
__device__ __forceinline__ V4F MmaTile(const uint4& w, const uint4& x,
                                       V4F acc) {
  acc = Mma<T>((V2I){(int)w.x, (int)w.y}, (V2I){(int)x.x, (int)x.y}, acc);
  return Mma<T>((V2I){(int)w.z, (int)w.w}, (V2I){(int)x.z, (int)x.w}, acc);
}

__device__ __forceinline__ uint4 Zero4() { return make_uint4(0u, 0u, 0u, 0u); }

__device__ __forceinline__ uint4 Masked(uint4 v, uint32_t keep) {
  return make_uint4(v.x & keep, v.y & keep, v.z & keep, v.w & keep);
}

// LDS slot of the A fragment for accumulator column (4 * c + kg) of the lane's
// row.
__device__ __forceinline__ int Fragment(int slot, int kg, int c) {
  return slot * 16 + ((4 * c + kg) ^ slot);
}

// One work item of a wavefront: 16 features (w13) or 32 output rows (w2) of one
// expert block. Besides the stream's inputs it carries what the epilogue stores
// need: the pair of each accumulator column (4 * r + kg) and, for w2, that
// pair's routing weight. All of it is fetched one item ahead, so no dependent
// load remains once the accumulators are complete.
struct Item {
  int block, group, expert, pair, q[4];
  float w[4];
  bool active;
};

// Publish one k step to LDS: v[band][g] is row (g * 4 + kg), unit (step * 16 +
// slot).
__device__ __forceinline__ void Publish(uint4* lds, const uint4 (&v)[2][4],
                                        int slot, int kg) {
#pragma unroll
  for (int b = 0; b < 2; ++b) {
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const int r = g * 4 + kg;
      lds[b * kBand + r * 16 + (slot ^ r)] = v[b][g];
    }
  }
}

// Four activation fragments of k step s for the lane's token slot: zero where
// `keep` is 0 and beyond k_total. `x` already includes the lane's k offset (kg
// * kUnit) and always points at a valid row.
__device__ __forceinline__ void LoadX4(uint4 (&xq)[4],
                                       const uint16_t* __restrict__ x, int s,
                                       int k_total, int kg, uint32_t keep) {
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int offset = s * kStep + c * 32;
    const bool in_row = offset + kg * kUnit + kUnit <= k_total;
    // Beyond k_total a lane reads the last 16 bytes it can reach inside the row
    // instead (k_total is at least 32, so there is one for every lane) and is
    // masked below.
    const uint4 t = *reinterpret_cast<const uint4*>(
        x + (in_row ? offset : k_total - kg * kUnit - kUnit));
    xq[c] = Masked(t, keep & (in_row ? 0xffffffffu : 0u));
  }
}

// Weight loads of k step s. Rows and units beyond the matrix are zero.
// kUp: w13, whose second band holds the up rows (i + f); otherwise w2, whose
// second band holds the next 16 output rows.
template <bool kUp>
__device__ __forceinline__ void LoadWeights(uint4 (&v)[2][4],
                                            const uint16_t* __restrict__ w,
                                            int expert, int group,
                                            int rows_per_band, int k_total,
                                            int slot, int kg, int s) {
  if (expert < 0) {  // a block without an expert contributes zeros
#pragma unroll
    for (int b = 0; b < 2; ++b) {
#pragma unroll
      for (int g = 0; g < 4; ++g) v[b][g] = Zero4();
    }
    return;
  }
  constexpr int kTile = kUp ? kBlock : 2 * kBlock;  // rows a work item covers
  const int band_rows = kUp ? rows_per_band : kBlock;
  const uint16_t* base =
      w +
      static_cast<size_t>(expert) * (kUp ? 2 : 1) * rows_per_band * k_total +
      static_cast<size_t>(group * kTile) * k_total;
  const bool full =
      (group + 1) * kTile <= rows_per_band && (s + 1) * kStep <= k_total;
#pragma unroll
  for (int b = 0; b < 2; ++b) {
    const uint16_t* band = base + static_cast<size_t>(b * band_rows) * k_total;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const int row = g * 4 + kg, unit = s * 16 + slot;
      const uint16_t* p =
          band + static_cast<size_t>(row) * k_total + unit * kUnit;
      if (full) {
        v[b][g] = *reinterpret_cast<const uint4*>(p);
      } else {
        const bool ok =
            group * kTile + (kUp ? 0 : b * kBlock) + row < rows_per_band &&
            (unit + 1) * kUnit <= k_total;
        v[b][g] = ok ? *reinterpret_cast<const uint4*>(p) : Zero4();
      }
    }
  }
}

// act[pair, f] = silu(x . w13[e, f, :]) * (x . w13[e, i + f, :])
template <typename T>
__global__ __launch_bounds__(kLanes) void W13Kernel(
    const uint16_t* __restrict__ hidden, const uint16_t* __restrict__ w13,
    uint16_t* __restrict__ act, const int32_t* __restrict__ sorted,
    const int32_t* __restrict__ experts, const int32_t* __restrict__ total,
    int tokens, int topk, int h, int i, int items, int groups) {
  __shared__ __align__(16) uint4 lds[2 * kBand];
  const int slot = threadIdx.x & 15, kg = threadIdx.x >> 4;
  const int steps = (h + kStep - 1) / kStep, pairs = tokens * topk;
  const int padded = total[0];

  auto describe = [&](int item) {
    Item it{item / groups, 0, -1, 0, {0, 0, 0, 0}, {0.f, 0.f, 0.f, 0.f}, false};
    it.group = item - it.block * groups;
    it.active = it.block * kBlock < padded;
    if (it.active) {
      it.expert = experts[it.block];
      it.pair = sorted[it.block * kBlock + slot];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        it.q[r] = sorted[it.block * kBlock + 4 * r + kg];
      }
    }
    return it;
  };
  auto load = [&](uint4(&v)[2][4], const Item& it, int s) {
    LoadWeights<true>(v, w13, it.expert, it.group, i, h, slot, kg, s);
  };

  // The metadata of an item is fetched one item ahead and carried over, and the
  // loads of its first k step are issued during the previous item.
  int item = blockIdx.x;
  Item cur = item < items ? describe(item) : Item{};
  bool pending = false;  // v and xq already hold k step 0 of the current item
  while (item < items) {
    const int next = item + gridDim.x;
    Item fut{};
    if (next < items) fut = describe(next);
    if (!cur.active) {  // block slots beyond the padded rows are empty
      item = next;
      cur = fut;
      pending = false;
      continue;
    }
    const bool x_ok =
        cur.expert >= 0 && cur.pair < pairs;  // padding holds the sentinel
    const uint16_t* x =
        hidden +
        static_cast<size_t>(cur.pair < pairs ? cur.pair / topk : 0) * h +
        kg * kUnit;
    const uint32_t keep = x_ok ? 0xffffffffu : 0u;
    uint4 v[2][4], xq[4], xn[4];
    if (!pending) {
      load(v, cur, 0);
      LoadX4(xq, x, 0, h, kg, keep);
    }
    V4F acc[2];
    acc[0] = acc[1] = (V4F){0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < steps; ++s) {
      __syncthreads();  // LDS reads of the previous step are complete
      Publish(lds, v, slot, kg);
      __syncthreads();
      const bool more = s + 1 < steps;
      if (more) {
        load(v, cur, s + 1);
        LoadX4(xn, x, s + 1, h, kg, keep);
      } else if (next < items) {
        load(v, fut, 0);
        LoadX4(xn,
               hidden +
                   static_cast<size_t>(fut.pair < pairs ? fut.pair / topk : 0) *
                       h +
                   kg * kUnit,
               0, h, kg,
               fut.expert >= 0 && fut.pair < pairs ? 0xffffffffu : 0u);
      }
      uint4 wg[4], wu[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        wg[c] = lds[Fragment(slot, kg, c)];
        wu[c] = lds[kBand + Fragment(slot, kg, c)];
      }
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        acc[0] = MmaTile<T>(wg[c], xq[c], acc[0]);
        acc[1] = MmaTile<T>(wu[c], xq[c], acc[1]);
      }
      if (more || next < items) {
#pragma unroll
        for (int c = 0; c < 4; ++c) xq[c] = xn[c];
      }
    }
    const int row = cur.group * kBlock + slot;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int q = cur.q[r];
      if (q < pairs) {
        const float g = acc[0][r];
        const uint16_t value = FromFloat<T>(g / (1.0f + expf(-g)) * acc[1][r]);
        if (row < i) act[static_cast<size_t>(q) * i + row] = value;
      }
    }
    item = next;
    cur = fut;
    pending = next < items;
  }
}

// pair[pair, h] = weight[pair] * (act[pair] . w2[expert]^T)
template <typename T>
__global__ __launch_bounds__(kLanes) void W2Kernel(
    const uint16_t* __restrict__ act, const uint16_t* __restrict__ w2,
    float* __restrict__ pair_out, const float* __restrict__ weights,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ experts,
    const int32_t* __restrict__ total, int tokens, int topk, int h, int i,
    int items, int groups) {
  __shared__ __align__(16) uint4 lds[2 * kBand];
  const int slot = threadIdx.x & 15, kg = threadIdx.x >> 4;
  const int steps = (i + kStep - 1) / kStep, pairs = tokens * topk;
  const int padded = total[0];

  auto describe = [&](int item) {
    Item it{item / groups, 0, -1, 0, {0, 0, 0, 0}, {0.f, 0.f, 0.f, 0.f}, false};
    it.group = item - it.block * groups;
    it.active = it.block * kBlock < padded;
    if (it.active) {
      it.expert = experts[it.block];
      it.pair = sorted[it.block * kBlock + slot];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        it.q[r] = sorted[it.block * kBlock + 4 * r + kg];
        it.w[r] = it.q[r] < pairs ? weights[it.q[r]] : 0.f;
      }
    }
    return it;
  };
  auto load = [&](uint4(&v)[2][4], const Item& it, int s) {
    LoadWeights<false>(v, w2, it.expert, it.group, h, i, slot, kg, s);
  };

  // See W13Kernel for the pipelining across items.
  int item = blockIdx.x;
  Item cur = item < items ? describe(item) : Item{};
  bool pending = false;
  while (item < items) {
    const int next = item + gridDim.x;
    Item fut{};
    if (next < items) fut = describe(next);
    if (!cur.active) {
      item = next;
      cur = fut;
      pending = false;
      continue;
    }
    const bool x_ok = cur.expert >= 0 && cur.pair < pairs;
    const uint16_t* x =
        act + static_cast<size_t>(cur.pair < pairs ? cur.pair : 0) * i +
        kg * kUnit;
    const uint32_t keep = x_ok ? 0xffffffffu : 0u;
    uint4 v[2][4], xq[4], xn[4];
    if (!pending) {
      load(v, cur, 0);
      LoadX4(xq, x, 0, i, kg, keep);
    }
    V4F acc[2];
    acc[0] = acc[1] = (V4F){0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < steps; ++s) {
      __syncthreads();
      Publish(lds, v, slot, kg);
      __syncthreads();
      const bool more = s + 1 < steps;
      if (more) {
        load(v, cur, s + 1);
        LoadX4(xn, x, s + 1, i, kg, keep);
      } else if (next < items) {
        load(v, fut, 0);
        LoadX4(xn,
               act + static_cast<size_t>(fut.pair < pairs ? fut.pair : 0) * i +
                   kg * kUnit,
               0, i, kg,
               fut.expert >= 0 && fut.pair < pairs ? 0xffffffffu : 0u);
      }
      uint4 wt[2][4];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        wt[0][c] = lds[Fragment(slot, kg, c)];
        wt[1][c] = lds[kBand + Fragment(slot, kg, c)];
      }
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        acc[0] = MmaTile<T>(wt[0][c], xq[c], acc[0]);
        acc[1] = MmaTile<T>(wt[1][c], xq[c], acc[1]);
      }
      if (more || next < items) {
#pragma unroll
        for (int c = 0; c < 4; ++c) xq[c] = xn[c];
      }
    }
    const int row = cur.group * 2 * kBlock + slot;
#pragma unroll
    for (int b = 0; b < 2; ++b) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int q = cur.q[r];
        if (q < pairs) {
          const float value = acc[b][r] * cur.w[r];
          if (row + b * kBlock < h) {
            pair_out[static_cast<size_t>(q) * h + row + b * kBlock] = value;
          }
        }
      }
    }
    item = next;
    cur = fut;
    pending = next < items;
  }
}

// output[token] = sum over the token's active routes (the pair_out rows already
// carry their routing weight)
template <typename T>
__global__ __launch_bounds__(256) void CombineKernel(
    const float* __restrict__ pair_out, const int32_t* __restrict__ ids,
    uint16_t* __restrict__ out, int tokens, int topk, int h, int num_experts) {
  const int token = blockIdx.x;
  for (int c = threadIdx.x * 4; c < h; c += blockDim.x * 4) {
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s = 0; s < topk; ++s) {
      const int e = ids[token * topk + s];
      if (e < 0 || e >= num_experts) continue;
      const float4 v = *reinterpret_cast<const float4*>(
          pair_out + static_cast<size_t>(token * topk + s) * h + c);
      sum.x += v.x;
      sum.y += v.y;
      sum.z += v.z;
      sum.w += v.w;
    }
    const uint2 packed =
        make_uint2(FromFloat<T>(sum.x) | (FromFloat<T>(sum.y) << 16),
                   FromFloat<T>(sum.z) | (FromFloat<T>(sum.w) << 16));
    *reinterpret_cast<uint2*>(out + static_cast<size_t>(token) * h + c) =
        packed;
  }
}

}  // namespace infini::ops::moe_fused_dense_detail
#endif  // INFINI_OPS_HYGON_MOE_FUSED_DENSE_KERNEL_CUH_
