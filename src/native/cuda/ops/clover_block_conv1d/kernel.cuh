#ifndef INFINI_OPS_CUDA_CLOVER_BLOCK_CONV1D_KERNEL_CUH_
#define INFINI_OPS_CUDA_CLOVER_BLOCK_CONV1D_KERNEL_CUH_

#include <cstdint>

#include "native/cuda/caster.cuh"

namespace infini::ops {

// Shared-layer SIMT kernel for CloverBlockConv1d, mirroring the Triton
// reference (SGLang lineage). Per (request, token, channel):
//   left part  : sum over 4 taps of input/state history . weight[c,0,tap]
//   right part : sum over 3 taps of future-in-block input . right_weight[c,0,d-1]
//   out        = silu(left + right), zeroed for inactive rows.
// The state pool row for a request holds its last 3 pre-block values
// (clean prefix); only rows whose `valid` flag is set contribute.
// Grid: x covers channels in BLOCK_C tiles, y iterates (request, tile).
template <Device::Type kDevice, typename Data, typename Index>
__global__ void CloverBlockConv1dKernel(
    const Data* __restrict__ input, const Data* __restrict__ state,
    const Index* __restrict__ indices, const Data* __restrict__ weight,
    const Data* __restrict__ right_weight, const int8_t* __restrict__ valid,
    Data* __restrict__ out, int64_t channels, int block_size, int num_slots,
    int64_t stride_t, int64_t stride_c, int64_t state_slot_stride,
    int64_t state_c_stride, int64_t state_w_stride, int64_t weight_c_stride,
    int64_t weight_w_stride, int64_t right_c_stride, int64_t right_w_stride,
    int64_t out_t_stride, int64_t out_c_stride, int token_tile_t,
    int num_token_tiles) {
  const int64_t ch = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     static_cast<int64_t>(threadIdx.x);
  if (ch >= channels) return;

  const int token_program = static_cast<int>(blockIdx.y);
  const int request = token_program / num_token_tiles;
  const int tile = token_program % num_token_tiles;

  const auto cache_index = indices[request];
  const bool request_active = cache_index >= 0 && cache_index < num_slots;
  const auto safe_slot = cache_index >= 0 ? cache_index : 0;

  const int token_base = tile * token_tile_t;
  for (int t_in_tile = 0; t_in_tile < token_tile_t; ++t_in_tile) {
    const int t = token_base + t_in_tile;
    if (t >= block_size) break;
    const int64_t token = static_cast<int64_t>(request) * block_size + t;
    const bool row_active =
        request_active && (valid[token] != 0);
    if (!row_active) {
      out[token * out_t_stride + ch * out_c_stride] = Data(0);
      continue;
    }

    float acc = 0.0f;
    // Left/history taps (WIDTH = 4): t-3 .. t come from the block input;
    // earlier positions come from the clean state row (3 slots).
    for (int tap = 0; tap < 4; ++tap) {
      const int rel = t + tap - 3;
      float value;
      if (rel >= 0) {
        const int64_t src = static_cast<int64_t>(request) * block_size + rel;
        const bool src_active = valid[src] != 0;
        value = src_active
                    ? Caster<kDevice>::template Cast<float>(
                          input[src * stride_t + ch * stride_c])
                    : 0.0f;
      } else {
        const int64_t w = rel + 3;  // history position 0..2
        value = Caster<kDevice>::template Cast<float>(
            state[static_cast<int64_t>(safe_slot) * state_slot_stride +
                  ch * state_c_stride + w * state_w_stride]);
      }
      acc += value * Caster<kDevice>::template Cast<float>(
                        weight[ch * weight_c_stride + tap * weight_w_stride]);
    }
    // Right/symmetric taps: future tokens inside the same block.
    for (int distance = 1; distance < 4; ++distance) {
      const int right = t + distance;
      if (right < block_size) {
        const int64_t src = static_cast<int64_t>(request) * block_size + right;
        const bool src_active = valid[src] != 0;
        const float value =
            src_active
                ? Caster<kDevice>::template Cast<float>(
                      input[src * stride_t + ch * stride_c])
                : 0.0f;
        acc += value * Caster<kDevice>::template Cast<float>(
                           right_weight[ch * right_c_stride +
                                        (distance - 1) * right_w_stride]);
      }
    }
    const float sigmoid = 1.0f / (1.0f + expf(-acc));
    const float result = acc * sigmoid;
    out[token * out_t_stride + ch * out_c_stride] =
        Caster<kDevice>::template Cast<Data>(result);
  }
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_BLOCK_CONV1D_KERNEL_CUH_
