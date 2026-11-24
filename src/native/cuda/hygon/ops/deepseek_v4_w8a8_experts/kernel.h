#ifndef INFINI_OPS_HYGON_DEEPSEEK_V4_W8A8_EXPERTS_H_
#define INFINI_OPS_HYGON_DEEPSEEK_V4_W8A8_EXPERTS_H_

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "base/deepseek_v4_w8a8_experts.h"
#include "kernel_v2.cuh"
#include "math.cuh"
#include "native/cuda/hygon/dynamic_int8_quant.cuh"

namespace infini::ops {
namespace deepseek_v4_w8a8_detail {

using hygon_dynamic_int8::kThreads;
using hygon_dynamic_int8::Quantize;

template <typename T>
__global__ void GateUp(const int8_t* x, const float* input_scale,
                       const int64_t* indices, const int8_t* w,
                       const float* scale, float* activated, size_t routes,
                       size_t top_k, size_t experts, size_t hidden,
                       size_t inner, float limit, bool routed) {
  __shared__ int32_t gate_sum[kThreads];
  __shared__ int32_t up_sum[kThreads];
  const unsigned lane = threadIdx.x;
  for (size_t index = blockIdx.x; index < routes * inner; index += gridDim.x) {
    const size_t route = index / inner;
    const auto expert = indices[route];
    if (expert < 0 || static_cast<size_t>(expert) >= experts) {
      if (lane == 0) activated[index] = 0.0f;
      continue;
    }
    const size_t gate =
        static_cast<size_t>(expert) * (2 * inner) + index % inner;
    const size_t up = gate + inner;
    const auto* row = x + (route / top_k) * hidden;
    int32_t g = 0, u = 0;
    for (size_t i = lane; i < hidden; i += kThreads) {
      g += static_cast<int32_t>(row[i]) * w[gate * hidden + i];
      u += static_cast<int32_t>(row[i]) * w[up * hidden + i];
    }
    gate_sum[lane] = g;
    up_sum[lane] = u;
    __syncthreads();
    for (unsigned width = kThreads / 2; width; width /= 2) {
      if (lane < width) {
        gate_sum[lane] += gate_sum[lane + width];
        up_sum[lane] += up_sum[lane + width];
      }
      __syncthreads();
    }
    if (lane == 0) {
      using Convert = Caster<Device::Type::kHygon>;
      const float token_scale = input_scale[route / top_k];
      const auto cast_gate = Convert::template Cast<T>(
          ScaledDot(gate_sum[0], token_scale, scale[gate],
                    routed || routes / top_k > 16));
      const auto cast_up = Convert::template Cast<T>(ScaledDot(
          up_sum[0], token_scale, scale[up], routed || routes / top_k > 16));
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
    __syncthreads();
  }
}

template <typename T>
__global__ void Down(const int8_t* activated, const float* activation_scale,
                     const int64_t* indices, const float* weights,
                     const int8_t* w, const float* scale, T* out, size_t tokens,
                     size_t top_k, size_t experts, size_t hidden, size_t inner,
                     bool routed, float routed_scale) {
  __shared__ int32_t sum[kThreads];
  const unsigned lane = threadIdx.x;
  for (size_t index = blockIdx.x; index < tokens * hidden; index += gridDim.x) {
    const size_t token = index / hidden;
    const size_t channel = index % hidden;
    float output = 0.0f;
    for (size_t k = 0; k < top_k; ++k) {
      const size_t route = token * top_k + k;
      const auto expert = indices[route];
      if (expert < 0 || static_cast<size_t>(expert) >= experts) continue;
      const size_t channel_idx = static_cast<size_t>(expert) * hidden + channel;
      int32_t local = 0;
      for (size_t i = lane; i < inner; i += kThreads) {
        local += static_cast<int32_t>(activated[route * inner + i]) *
                 w[channel_idx * inner + i];
      }
      sum[lane] = local;
      __syncthreads();
      for (unsigned width = kThreads / 2; width; width /= 2) {
        if (lane < width) sum[lane] += sum[lane + width];
        __syncthreads();
      }
      if (lane == 0) {
        using Convert = Caster<Device::Type::kHygon>;
        const float token_scale = activation_scale[route],
                    weight_scale = scale[channel_idx];
        if (routed) {
          const float weighted =
              ScaledDot(sum[0], token_scale, weight_scale, true) *
              weights[route];
          output += Convert::template Cast<float>(
              Convert::template Cast<T>(weighted));
        } else {
          output += ScaledDot(sum[0], token_scale, weight_scale, tokens > 16) *
                    weights[route];
        }
      }
      __syncthreads();
    }
    if (lane == 0) {
      out[index] =
          Caster<Device::Type::kHygon>::template Cast<T>(output * routed_scale);
    }
  }
}

}  // namespace deepseek_v4_w8a8_detail

template <>
class Operator<DeepseekV4W8a8Experts, Device::Type::kHygon>
    : public DeepseekV4W8a8Experts {
 public:
  static bool supports(const Tensor x, const Tensor weights,
                       const Tensor indices, const Tensor gate_up,
                       const Tensor gate_up_scale, const Tensor down,
                       const Tensor down_scale, double limit,
                       const std::optional<double> routed_scale,
                       const Tensor out) {
    if (x.device().type() != Device::Type::kHygon || x.ndim() != 2 ||
        weights.ndim() != 2 || gate_up.ndim() != 3 || down.ndim() != 3 ||
        out.shape() != x.shape() || out.dtype() != x.dtype() ||
        (x.dtype() != DataType::kBFloat16 && x.dtype() != DataType::kFloat32) ||
        weights.dtype() != DataType::kFloat32 ||
        indices.dtype() != DataType::kInt64 ||
        weights.shape() != indices.shape() || weights.size(0) != x.size(0) ||
        gate_up.dtype() != DataType::kInt8 || down.dtype() != DataType::kInt8 ||
        gate_up_scale.dtype() != DataType::kFloat32 ||
        down_scale.dtype() != DataType::kFloat32 || !std::isfinite(limit) ||
        limit <= 0.0 || limit > std::numeric_limits<float>::max() ||
        (routed_scale &&
         (!std::isfinite(*routed_scale) || *routed_scale <= 0.0 ||
          *routed_scale > std::numeric_limits<float>::max())))
      return false;

    const auto experts = down.size(0), hidden = x.size(1), inner = down.size(2);
    const auto size_limit = std::numeric_limits<size_t>::max();
    if (!experts || !hidden || !inner || !x.size(0) || !weights.size(1) ||
        hidden > kMaxDotWidth || inner > kMaxDotWidth ||
        x.size(0) > size_limit / (hidden + sizeof(float)) / 2 ||
        x.size(0) > size_limit / weights.size(1) /
                        (inner * (sizeof(float) + 1) + sizeof(float)) / 2 ||
        gate_up.shape() != Tensor::Shape{experts, 2 * inner, hidden} ||
        down.shape() != Tensor::Shape{experts, hidden, inner} ||
        gate_up_scale.shape() != Tensor::Shape{experts, 2 * inner, 1} ||
        down_scale.shape() != Tensor::Shape{experts, hidden, 1})
      return false;
    for (const auto* t : {&x, &weights, &indices, &gate_up, &gate_up_scale,
                          &down, &down_scale, &out}) {
      if (t->device() != x.device() || !t->IsContiguous()) return false;
    }
    return true;
  }

  Operator(const Tensor x, const Tensor weights, const Tensor indices,
           const Tensor gate_up, const Tensor gate_up_scale, const Tensor down,
           const Tensor down_scale, double limit,
           const std::optional<double> routed_scale, Tensor out)
      : DeepseekV4W8a8Experts(x, weights, indices, gate_up, gate_up_scale, down,
                              down_scale, limit, routed_scale, out) {
    if (!supports(x, weights, indices, gate_up, gate_up_scale, down, down_scale,
                  limit, routed_scale, out))
      throw std::invalid_argument(
          "DeepSeek V4 W8A8 expert input is unsupported");
    workspace_bytes_ = workspace_bytes(x, weights, down.size(2));
  }

  std::size_t workspace_size_in_bytes() const override {
    return workspace_bytes_;
  }
  bool capture_safe() const override { return true; }

  void operator()(const Tensor x, const Tensor weights, const Tensor indices,
                  const Tensor gate_up, const Tensor gate_up_scale,
                  const Tensor down, const Tensor down_scale, double limit,
                  const std::optional<double> routed_scale,
                  Tensor out) const override {
    if (!supports(x, weights, indices, gate_up, gate_up_scale, down, down_scale,
                  limit, routed_scale, out) ||
        workspace_bytes(x, weights, down.size(2)) != workspace_bytes_ ||
        workspace_size_in_bytes_ < workspace_bytes_ || !workspace_)
      throw std::invalid_argument(
          "DeepSeek V4 W8A8 expert inputs or workspace changed");
    if (x.dtype() == DataType::kFloat32) {
      launch<float>(x, weights, indices, gate_up, gate_up_scale, down,
                    down_scale, limit, routed_scale, out);
    } else {
      launch<TypeMapType<Device::Type::kHygon, DataType::kBFloat16>>(
          x, weights, indices, gate_up, gate_up_scale, down, down_scale, limit,
          routed_scale, out);
    }
  }

 private:
  static constexpr size_t kMaxDotWidth =
      std::numeric_limits<int32_t>::max() / (127 * 128);
  size_t workspace_bytes_{};

  static size_t workspace_bytes(const Tensor x, const Tensor weights,
                                size_t inner) {
    const size_t tokens = x.size(0), routes = weights.numel();
    return (routes * inner + tokens + routes) * sizeof(float) +
           tokens * x.size(1) + routes * inner;
  }

  template <class T>
  void launch(const Tensor x, const Tensor weights, const Tensor indices,
              const Tensor gate_up, const Tensor gate_up_scale,
              const Tensor down, const Tensor down_scale, double limit,
              const std::optional<double> routed_scale, Tensor out) const {
    using namespace deepseek_v4_w8a8_detail;
    const auto stream = static_cast<cudaStream_t>(stream_);
    const auto routes = weights.numel(), inner = down.size(2);
    const auto* ids = static_cast<const int64_t*>(indices.data());
    const auto tokens = x.size(0), hidden = x.size(1);
    auto* activated = static_cast<float*>(workspace_);
    auto* input_scale = activated + routes * inner;
    auto* activation_scale = input_scale + tokens;
    auto* q_input = reinterpret_cast<int8_t*>(activation_scale + routes);
    auto* q_activated = q_input + tokens * hidden;
    const auto grid = [](size_t rows) {
      return static_cast<unsigned>(rows < 65535 ? rows : 65535);
    };
    // Keep the scalar path for packed rows with insufficient alignment.
    const bool packed = hidden % 4 == 0 && inner % 4 == 0 &&
                        reinterpret_cast<uintptr_t>(q_input) % 4 == 0 &&
                        reinterpret_cast<uintptr_t>(q_activated) % 4 == 0 &&
                        reinterpret_cast<uintptr_t>(gate_up.data()) % 4 == 0 &&
                        reinterpret_cast<uintptr_t>(down.data()) % 4 == 0;
    if (packed) {
      expert_wave::Quant<T><<<grid(tokens), 256, 0, stream>>>(
          static_cast<const T*>(x.data()), q_input, input_scale, tokens,
          hidden);
    } else {
      Quantize<T, true><<<grid(tokens), kThreads, 0, stream>>>(
          static_cast<const T*>(x.data()), q_input, input_scale, tokens,
          hidden);
    }
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("DeepSeek V4 input quantization launch failed");
    if (packed) {
      expert_wave::Gate<T><<<grid((routes * inner + 3) / 4), 256, 0, stream>>>(
          q_input, input_scale, ids, static_cast<const int8_t*>(gate_up.data()),
          static_cast<const float*>(gate_up_scale.data()), activated, routes,
          weights.size(1), down.size(0), hidden, inner,
          static_cast<float>(limit), routed_scale.has_value());
    } else {
      GateUp<T><<<grid(routes * inner), kThreads, 0, stream>>>(
          q_input, input_scale, ids, static_cast<const int8_t*>(gate_up.data()),
          static_cast<const float*>(gate_up_scale.data()), activated, routes,
          weights.size(1), down.size(0), hidden, inner,
          static_cast<float>(limit), routed_scale.has_value());
    }
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("DeepSeek V4 gate/up launch failed");
    if (packed) {
      expert_wave::Quant<float><<<grid(routes), 256, 0, stream>>>(
          activated, q_activated, activation_scale, routes, inner);
    } else {
      Quantize<float, true><<<grid(routes), kThreads, 0, stream>>>(
          activated, q_activated, activation_scale, routes, inner);
    }
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error(
          "DeepSeek V4 activation quantization launch failed");
    if (packed) {
      expert_wave::Down<T><<<grid((out.numel() + 3) / 4), 256, 0, stream>>>(
          q_activated, activation_scale, ids,
          static_cast<const float*>(weights.data()),
          static_cast<const int8_t*>(down.data()),
          static_cast<const float*>(down_scale.data()),
          static_cast<T*>(out.data()), tokens, weights.size(1), down.size(0),
          hidden, inner, routed_scale.has_value(),
          static_cast<float>(routed_scale.value_or(1.0)));
    } else {
      Down<T><<<grid(out.numel()), kThreads, 0, stream>>>(
          q_activated, activation_scale, ids,
          static_cast<const float*>(weights.data()),
          static_cast<const int8_t*>(down.data()),
          static_cast<const float*>(down_scale.data()),
          static_cast<T*>(out.data()), tokens, weights.size(1), down.size(0),
          hidden, inner, routed_scale.has_value(),
          static_cast<float>(routed_scale.value_or(1.0)));
    }
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("DeepSeek V4 down launch failed");
  }
};

}  // namespace infini::ops

#endif
