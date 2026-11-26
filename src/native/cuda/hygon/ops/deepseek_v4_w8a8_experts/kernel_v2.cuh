#pragma once
#include "math.cuh"
#include "native/cuda/hygon/dynamic_int8_quant.cuh"
namespace infini::ops::expert_wave {
__device__ __forceinline__ int DotWord(int a, int b, int sum) {
#if defined(__gfx936__)
  return __builtin_amdgcn_sdot4(a, b, sum, false);
#else
  for (int j = 0; j < 4; ++j)
    sum +=
        static_cast<int8_t>(a >> (8 * j)) * static_cast<int8_t>(b >> (8 * j));
  return sum;
#endif
}
__device__ __forceinline__ int Reduce(int sum) {
  for (int delta = 32; delta > 0; delta >>= 1)
    sum += __shfl_down_sync(0xffffffffffffffffull, sum, delta, 64);
  return sum;
}
template <class T>
__global__ void Quant(const T* input, int8_t* q, float* scales, size_t rows,
                      size_t width) {
  const unsigned lane = threadIdx.x & 63, wave = threadIdx.x / 64;
  __shared__ float maxima[4];
  for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    float maximum = 0.0f;
    for (size_t j = threadIdx.x; j < width; j += blockDim.x)
      maximum = fmaxf(maximum,
                      fabsf(Caster<Device::Type::kHygon>::template Cast<float>(
                          input[row * width + j])));
    for (int delta = 32; delta > 0; delta >>= 1)
      maximum = fmaxf(
          maximum, __shfl_xor_sync(0xffffffffffffffffull, maximum, delta, 64));
    if (lane == 0) maxima[wave] = maximum;
    __syncthreads();
    for (unsigned i = 0; i < blockDim.x / 64; ++i)
      maximum = fmaxf(maximum, maxima[i]);
    maximum = fmaxf(maximum, 1.0e-10f);
    const float scale = maximum / 127.0f, inverse = 127.0f / maximum;
    if (threadIdx.x == 0) scales[row] = scale;
    for (size_t j = threadIdx.x; j < width; j += blockDim.x) {
      const int value =
          __float2int_rn(Caster<Device::Type::kHygon>::template Cast<float>(
                             input[row * width + j]) *
                         inverse);
      q[row * width + j] = static_cast<int8_t>(min(max(value, -127), 127));
    }
    __syncthreads();
  }
}
template <class T>
__global__ void Gate(const int8_t* x, const float* input_scale,
                     const int64_t* ids, const int8_t* w, const float* scale,
                     float* activated, size_t routes, size_t top_k,
                     size_t experts, size_t hidden, size_t inner, float limit,
                     bool routed) {
  const unsigned lane = threadIdx.x & 63, wave = threadIdx.x / 64;
  const size_t first = static_cast<size_t>(blockIdx.x) * 4 + wave;
  for (size_t index = first; index < routes * inner;
       index += static_cast<size_t>(gridDim.x) * 4) {
    const size_t route = index / inner;
    const auto expert = ids[route];
    if (expert < 0 || static_cast<size_t>(expert) >= experts) {
      if (lane == 0) activated[index] = 0.0f;
      continue;
    }
    const size_t gate =
                     static_cast<size_t>(expert) * (2 * inner) + index % inner,
                 up = gate + inner;
    const auto* input =
        reinterpret_cast<const int*>(x + (route / top_k) * hidden);
    const auto* gw = reinterpret_cast<const int*>(w + gate * hidden);
    const auto* uw = reinterpret_cast<const int*>(w + up * hidden);
    int g = 0, u = 0;
    for (size_t j = lane; j < hidden / 4; j += 64) {
      const int value = __ldg(input + j);
      g = DotWord(value, __ldg(gw + j), g);
      u = DotWord(value, __ldg(uw + j), u);
    }
    g = Reduce(g);
    u = Reduce(u);
    if (lane == 0) {
      using Convert = Caster<Device::Type::kHygon>;
      using deepseek_v4_w8a8_detail::ScaledDot;
      const float token_scale = input_scale[route / top_k];
      const auto cast_gate = Convert::template Cast<T>(ScaledDot(
          g, token_scale, scale[gate], routed || routes / top_k > 16));
      const auto cast_up = Convert::template Cast<T>(
          ScaledDot(u, token_scale, scale[up], routed || routes / top_k > 16));
      float clipped_gate =
          fminf(Convert::template Cast<float>(cast_gate), limit);
      if (routed) clipped_gate = fmaxf(clipped_gate, -limit);
      const float clipped_up =
          fminf(fmaxf(Convert::template Cast<float>(cast_up), -limit), limit);
      float gate_activation = clipped_gate / (1.0f + expf(-clipped_gate));
      if (routed)
        gate_activation = Convert::template Cast<float>(
            Convert::template Cast<T>(gate_activation));
      const float value = gate_activation * clipped_up;
      activated[index] =
          Convert::template Cast<float>(Convert::template Cast<T>(value));
    }
  }
}
template <class T>
__global__ void Down(const int8_t* activated, const float* activation_scale,
                     const int64_t* ids, const float* weights, const int8_t* w,
                     const float* scale, T* out, size_t tokens, size_t top_k,
                     size_t experts, size_t hidden, size_t inner, bool routed,
                     float routed_scale) {
  const unsigned lane = threadIdx.x & 63, wave = threadIdx.x / 64;
  const size_t first = static_cast<size_t>(blockIdx.x) * 4 + wave;
  for (size_t index = first; index < tokens * hidden;
       index += static_cast<size_t>(gridDim.x) * 4) {
    const size_t token = index / hidden, channel = index % hidden;
    float output = 0.0f;
    for (size_t k = 0; k < top_k; ++k) {
      const size_t route = token * top_k + k;
      const auto expert = ids[route];
      if (expert < 0 || static_cast<size_t>(expert) >= experts) continue;
      const size_t ci = static_cast<size_t>(expert) * hidden + channel;
      const auto* input =
          reinterpret_cast<const int*>(activated + route * inner);
      const auto* wp = reinterpret_cast<const int*>(w + ci * inner);
      int sum = 0;
      for (size_t j = lane; j < inner / 4; j += 64)
        sum = DotWord(__ldg(input + j), __ldg(wp + j), sum);
      sum = Reduce(sum);
      if (lane == 0) {
        using Convert = Caster<Device::Type::kHygon>;
        using deepseek_v4_w8a8_detail::ScaledDot;
        const float token_scale = activation_scale[route],
                    weight_scale = scale[ci];
        if (routed) {
          const float weighted =
              ScaledDot(sum, token_scale, weight_scale, true) * weights[route];
          output += Convert::template Cast<float>(
              Convert::template Cast<T>(weighted));
        } else
          output += ScaledDot(sum, token_scale, weight_scale, tokens > 16) *
                    weights[route];
      }
    }
    if (lane == 0)
      out[index] =
          Caster<Device::Type::kHygon>::template Cast<T>(output * routed_scale);
  }
}
}  // namespace infini::ops::expert_wave
