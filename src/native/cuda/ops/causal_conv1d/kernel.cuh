#ifndef INFINI_OPS_CUDA_CAUSAL_CONV1D_KERNEL_CUH_
#define INFINI_OPS_CUDA_CAUSAL_CONV1D_KERNEL_CUH_

#include <cstddef>
#include <cstdint>

#include "data_type.h"

namespace infini::ops {
namespace causal_conv1d_detail {

struct Layout {
  ptrdiff_t strides[9][3]{};
  size_t tokens, channels, pool;
  bool cu_i64, initial_i64, final_i64;
};

__device__ inline int64_t index(const void* p, bool i64, int i, int fallback) {
  return p ? (i64 ? static_cast<const int64_t*>(p)[i]
                  : static_cast<int64_t>(static_cast<const int32_t*>(p)[i]))
           : fallback;
}

template <typename T>
__device__ inline float history(const T* state, const T* qkv, int64_t pos,
                                int64_t begin, int batch, int channel,
                                ptrdiff_t base, const Layout& p) {
  if (pos < 3) return static_cast<float>(state[base + pos * p.strides[2][2]]);
  const auto off = static_cast<ptrdiff_t>(batch) * p.strides[0][0] +
                   static_cast<ptrdiff_t>(begin + pos - 3) * p.strides[0][1] +
                   static_cast<ptrdiff_t>(channel) * p.strides[0][2];
  return static_cast<float>(qkv[off]);
}

// Port of baseline InfiniCore-0.2.9 causal_conv1d_k4_kernel. One thread per
// request/channel reads the entire sequence before committing its final state.
template <Device::Type kDevice, typename T>
__global__ void CausalConv1dKernel(const T* qkv, const T* weight, T* state,
                                   const T* bias, const void* cu,
                                   const void* initial,
                                   const void* final_indices, T* out,
                                   T* final_state, Layout p) {
  const int request = blockIdx.x, channel = blockIdx.y;
  if (static_cast<size_t>(channel) >= p.channels || threadIdx.x != 0) return;
  int64_t begin = 0, end = static_cast<int64_t>(p.tokens);
  int batch = request;
  if (cu) {
    begin = index(cu, p.cu_i64, request, 0);
    end = index(cu, p.cu_i64, request + 1, 0);
    batch = 0;
    // Preserve baseline device-value behavior: malformed range leaves the
    // corresponding output/state untouched. This is not host validation.
    if (begin < 0 || end < begin || end > static_cast<int64_t>(p.tokens))
      return;
  }
  const int64_t length = end - begin;
  const int64_t read = index(initial, p.initial_i64, request, request);
  const int64_t write = index(final_indices, p.final_i64, request, request);
  if (read < 0 || write < 0 || read >= static_cast<int64_t>(p.pool) ||
      (final_indices && write >= static_cast<int64_t>(p.pool)))
    return;
  const auto base = static_cast<ptrdiff_t>(read) * p.strides[2][0] +
                    static_cast<ptrdiff_t>(channel) * p.strides[2][1];
  const auto wb = static_cast<ptrdiff_t>(channel) * p.strides[1][0];
  for (int64_t t = 0; t < length; ++t) {
    float acc = static_cast<float>(weight[wb]) *
                    history(state, qkv, t, begin, batch, channel, base, p) +
                static_cast<float>(weight[wb + p.strides[1][2]]) *
                    history(state, qkv, t + 1, begin, batch, channel, base, p) +
                static_cast<float>(weight[wb + 2 * p.strides[1][2]]) *
                    history(state, qkv, t + 2, begin, batch, channel, base, p) +
                static_cast<float>(weight[wb + 3 * p.strides[1][2]]) *
                    history(state, qkv, t + 3, begin, batch, channel, base, p);
    if (bias)
      acc += static_cast<float>(
          bias[static_cast<ptrdiff_t>(channel) * p.strides[3][0]]);
    const auto off = static_cast<ptrdiff_t>(batch) * p.strides[7][0] +
                     static_cast<ptrdiff_t>(begin + t) * p.strides[7][1] +
                     static_cast<ptrdiff_t>(channel) * p.strides[7][2];
    out[off] = static_cast<T>(acc);
  }
  T* target = final_indices ? state : final_state;
  const int stride_index = final_indices ? 2 : 8;
  const auto target_base =
      static_cast<ptrdiff_t>(final_indices ? write : request) *
          p.strides[stride_index][0] +
      static_cast<ptrdiff_t>(channel) * p.strides[stride_index][1];
  // Same ordered commit as baseline. Exact initial/final alias is valid for
  // short sequences because copied history source positions lie ahead of
  // writes.
  target[target_base] = static_cast<T>(
      history(state, qkv, length, begin, batch, channel, base, p));
  target[target_base + p.strides[stride_index][2]] = static_cast<T>(
      history(state, qkv, length + 1, begin, batch, channel, base, p));
  target[target_base + 2 * p.strides[stride_index][2]] = static_cast<T>(
      history(state, qkv, length + 2, begin, batch, channel, base, p));
}

}  // namespace causal_conv1d_detail
}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CAUSAL_CONV1D_KERNEL_CUH_
