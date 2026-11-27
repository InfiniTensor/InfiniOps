#ifndef INFINI_OPS_CUDA_CLOVER_BLOCK_CONV1D_KERNEL_CUH_
#define INFINI_OPS_CUDA_CLOVER_BLOCK_CONV1D_KERNEL_CUH_

#include <cstdint>

#include "native/cuda/caster.cuh"

namespace infini::ops {

// Shared-layer SIMT kernel for CloverBlockConv1d, mirroring the Triton
// reference (SGLang lineage). Per (request, token, channel):
//   left part  : sum over 4 taps of input/state history . weight[c,0,tap]
//   right part : sum over 3 taps of future-in-block input .
//   right_weight[c,0,d-1] out        = silu(left + right), zeroed for inactive
//   rows.
// The state pool row for a request holds its last 3 pre-block values
// (clean prefix); only rows whose `valid` flag is set contribute.
//
// A thread owns one channel of a tile of kTileTokens consecutive tokens.
// What the tile reads is loaded first, in one batch: the block's rows from
// three tokens before the tile to three after it, with their validity
// flags, the weights and the request's slot. The validity flags only
// select; a row inside the block is read whatever its flag says. Only the
// state row, which needs the slot's index, waits for a second load.
// (Computing token by token, each with the loads it happens to need,
// strings several memory latencies one after another.)
//
// Grid: x covers channels in blocks of threads, y iterates (request, tile).
template <Device::Type kDevice, typename Data, typename Index, int kTileTokens>
__global__ void CloverBlockConv1dKernel(
    const Data* __restrict__ input, const Data* __restrict__ state,
    const Index* __restrict__ indices, const Data* __restrict__ weight,
    const Data* __restrict__ right_weight, const int8_t* __restrict__ valid,
    Data* __restrict__ out, int64_t channels, int block_size, int num_slots,
    int64_t stride_t, int64_t stride_c, int64_t state_slot_stride,
    int64_t state_c_stride, int64_t state_w_stride, int64_t weight_c_stride,
    int64_t weight_w_stride, int64_t right_c_stride, int64_t right_w_stride,
    int64_t out_t_stride, int64_t out_c_stride, int num_token_tiles) {
  constexpr int kWidth = 4;          // taps of the left part
  constexpr int kPast = kWidth - 1;  // tokens before the tile it may read
  constexpr int kRows = kTileTokens + 2 * kPast;

  const int64_t ch = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     static_cast<int64_t>(threadIdx.x);
  if (ch >= channels) return;

  const int token_program = static_cast<int>(blockIdx.y);
  const int request = token_program / num_token_tiles;
  const int first = (token_program % num_token_tiles) * kTileTokens;
  const int64_t base = static_cast<int64_t>(request) * block_size;
  const int64_t channel = ch * stride_c;

  const Index slot = indices[request];
  const bool request_active = slot >= 0 && slot < num_slots;

  // The block's rows first - 3 .. first + kTileTokens + 2. One past the
  // end of the block is zero, so is an invalid row.
  float row[kRows];
  unsigned live = 0;  // validity flags of the rows inside the block
#pragma unroll
  for (int i = 0; i < kRows; ++i) {
    const int token = first + i - kPast;
    const bool inside = token >= 0 && token < block_size;
    const bool on = inside && valid[base + token] != 0;
    const float value = inside ? Caster<kDevice>::template Cast<float>(
                                     input[(base + token) * stride_t + channel])
                               : 0.0f;
    live |= static_cast<unsigned>(on) << i;
    row[i] = on ? value : 0.0f;
  }
  float taps[kWidth], right_taps[kWidth - 1];
#pragma unroll
  for (int tap = 0; tap < kWidth; ++tap)
    taps[tap] = Caster<kDevice>::template Cast<float>(
        weight[ch * weight_c_stride + tap * weight_w_stride]);
#pragma unroll
  for (int distance = 1; distance < kWidth; ++distance)
    right_taps[distance - 1] = Caster<kDevice>::template Cast<float>(
        right_weight[ch * right_c_stride + (distance - 1) * right_w_stride]);
  // The rows before the block are the clean state, valid as they are. A
  // request without a slot in the pool reads none of it.
#pragma unroll
  for (int i = 0; i < kPast; ++i) {
    const int token = first + i - kPast;
    if (token < 0) {
      row[i] = request_active
                   ? Caster<kDevice>::template Cast<float>(
                         state[static_cast<int64_t>(slot) * state_slot_stride +
                               ch * state_c_stride +
                               (token + kPast) * state_w_stride])
                   : 0.0f;
    }
  }

  const int tokens =
      block_size - first < kTileTokens ? block_size - first : kTileTokens;
  for (int token = 0; token < tokens; ++token) {
    float acc = 0.0f;
#pragma unroll
    for (int tap = 0; tap < kWidth; ++tap) acc += row[token + tap] * taps[tap];
#pragma unroll
    for (int distance = 1; distance < kWidth; ++distance)
      acc += row[token + distance + kPast] * right_taps[distance - 1];
    const float sigmoid = 1.0f / (1.0f + expf(-acc));
    const bool active = request_active && ((live >> (token + kPast)) & 1u) != 0;
    out[(base + first + token) * out_t_stride + ch * out_c_stride] =
        Caster<kDevice>::template Cast<Data>(active ? acc * sigmoid : 0.0f);
  }
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_BLOCK_CONV1D_KERNEL_CUH_
