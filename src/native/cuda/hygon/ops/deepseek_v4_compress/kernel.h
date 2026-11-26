#ifndef INFINI_OPS_HYGON_DEEPSEEK_V4_COMPRESS_KERNEL_H_
#define INFINI_OPS_HYGON_DEEPSEEK_V4_COMPRESS_KERNEL_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "base/deepseek_v4_compress.h"
#include "native/cuda/hygon/caster.cuh"

namespace infini::ops {
namespace v4_compress_detail {

template<class T>
static __device__ void ReadSource(const T* projection, const float* state,
                                  const float* ape, int32_t source_id, size_t input_rows,
                                  size_t state_rows, size_t head_dim, size_t width,
                                  size_t position, size_t channel, bool overlap,
                                  bool checkpoint_ape, float& value, float& score) {
  value = 0.0f;
  score = -INFINITY;
  const size_t base = (overlap && position >= 4 ? head_dim : 0) + channel;
  if (source_id >= 0 && static_cast<size_t>(source_id) < input_rows) {
    using Convert = Caster<Device::Type::kHygon>;
    const auto* source = projection + static_cast<size_t>(source_id) * width;
    value = Convert::template Cast<float>(source[base]);
    score = Convert::template Cast<float>(source[base + width / 2]);
  } else if (source_id < -1 && static_cast<size_t>(-2LL - source_id) < state_rows) {
    const auto* source = state + static_cast<size_t>(-2LL - source_id) * width;
    value = source[base];
    score = source[base + width / 2];
  } else {
    return;
  }
  // The checkpoint stores the previous and current C4 bias halves side by side.
  const size_t ape_row = checkpoint_ape ? 2 * (position % 4) + (position >= 4) : position;
  score += ape[ape_row * head_dim + channel];
}

static __device__ float FastExp(float value) {
  return __builtin_amdgcn_exp2f(value * 1.4426950408889634f);
}

template<class T>
static __global__ void Pool(const T* projection, const float* ape,
                            const int32_t* sources, const float* state, T* out,
                            size_t input_rows, size_t state_rows, size_t output_rows,
                            size_t head_dim, size_t window, bool checkpoint_ape) {
  using Convert = Caster<Device::Type::kHygon>;
  const size_t width = (window == 8 ? 4 : 2) * head_dim;
  for (size_t row = blockIdx.x; row < output_rows; row += gridDim.x) {
    for (size_t channel = threadIdx.x; channel < head_dim; channel += blockDim.x) {
      const auto load = [&](size_t pos, float& value, float& score) {
        ReadSource(projection, state, ape, sources[row * window + pos], input_rows,
                   state_rows, head_dim, width, pos, channel, window == 8,
                   checkpoint_ape, value, score);
      };
      float maximum = -INFINITY;
      float sum = 0.0f;
      float total = 0.0f;
      if constexpr (std::is_same_v<T, float>) {
        // C4 uses one group. C128 merges sixteen groups of eight tokens.
        float group_maxima[16], group_sums[16], group_products[16];
        const size_t groups = window / 8;
        for (size_t group = 0; group < groups; ++group) {
          float values[8], scores[8], group_max = -INFINITY;
          for (size_t i = 0; i < 8; ++i) {
            load(group * 8 + i, values[i], scores[i]);
            group_max = fmaxf(group_max, scores[i]);
          }
          float group_sum = 0.0f, group_product = 0.0f;
          if (group_max != -INFINITY) {
            for (size_t i = 0; i < 8; ++i) {
              if (scores[i] == -INFINITY) continue;
              const float weight = FastExp(scores[i] - group_max);
              group_product = fmaf(values[i], weight, group_product);
              group_sum += weight;
            }
          }
          group_maxima[group] = group_max;
          group_sums[group] = group_sum;
          group_products[group] = group_product;
          maximum = fmaxf(maximum, group_max);
        }
        if (maximum != -INFINITY) {
          for (size_t group = 0; group < groups; ++group) {
            if (group_maxima[group] == -INFINITY) continue;
            const float weight = FastExp(group_maxima[group] - maximum);
            total = fmaf(group_products[group], weight, total);
            sum = fmaf(group_sums[group], weight, sum);
          }
        }
      } else {
        for (size_t pos = 0; pos < window; ++pos) {
          float value, score;
          load(pos, value, score);
          if (score == -INFINITY) continue;
          const float next_max = fmaxf(maximum, score);
          const float previous_weight = sum == 0.0f ? 0.0f : expf(maximum - next_max);
          const float weight = expf(score - next_max);
          total = total * previous_weight + weight * value;
          sum = sum * previous_weight + weight;
          maximum = next_max;
        }
      }
      out[row * head_dim + channel] =
          Convert::template Cast<T>(sum == 0.0f ? 0.0f : total / sum);
    }
  }
}

template<class T>
static __global__ void Update(const T* projection, const int32_t* slots, float* state,
                              size_t input_rows, size_t state_rows, size_t width) {
  for (size_t row = blockIdx.x; row < input_rows; row += gridDim.x) {
    const auto slot = slots[row];
    if (slot < 0 || static_cast<size_t>(slot) >= state_rows) continue;
    for (size_t col = threadIdx.x; col < width; col += blockDim.x) {
      state[static_cast<size_t>(slot) * width + col] =
          Caster<Device::Type::kHygon>::template Cast<float>(projection[row * width + col]);
    }
  }
}

}  // namespace v4_compress_detail

template <>
class Operator<DeepseekV4Compress, Device::Type::kHygon> : public DeepseekV4Compress {
 public:
  static bool supports(const Tensor projection, const Tensor ape, const Tensor sources,
                       const Tensor slots, const Tensor state, const Tensor out) {
    if (projection.device().type() != Device::Type::kHygon ||
        (projection.dtype() != DataType::kBFloat16 && projection.dtype() != DataType::kFloat32) ||
        ape.dtype() != DataType::kFloat32 ||
        sources.dtype() != DataType::kInt32 || slots.dtype() != DataType::kInt32 ||
        state.dtype() != DataType::kFloat32 || out.dtype() != projection.dtype() ||
        projection.ndim() != 2 || ape.ndim() != 2 || sources.ndim() != 2 ||
        state.ndim() != 2 || out.ndim() != 2) return false;
    const auto window = sources.size(1), dim = out.size(1);
    if ((window != 8 && window != 128) || (dim != 128 && dim != 512) ||
        (window == 128 && dim != 512) || !projection.size(0) || !state.size(0) ||
        projection.size(1) != dim * (window == 8 ? 4 : 2) ||
        state.size(1) != projection.size(1) || !sources.size(0) ||
        (ape.shape() != Tensor::Shape{window, dim} &&
         (window != 8 || ape.shape() != Tensor::Shape{4, 2 * dim})) ||
        out.size(0) != sources.size(0) ||
        slots.shape() != Tensor::Shape{projection.size(0)}) return false;
    for (const auto* t : {&projection, &ape, &sources, &slots, &state, &out}) {
      if (t->device() != projection.device() || !t->IsContiguous())
        return false;
    }
    return true;
  }

  Operator(const Tensor projection, const Tensor ape, const Tensor sources,
           const Tensor slots, Tensor state, Tensor out)
      : DeepseekV4Compress(projection, ape, sources, slots, state, out) {
    if (!supports(projection, ape, sources, slots, state, out))
      throw std::invalid_argument("DeepSeek V4 compressor inputs are unsupported");
  }

  bool capture_safe() const override { return true; }

  void operator()(const Tensor projection, const Tensor ape, const Tensor sources,
                  const Tensor slots, Tensor state, Tensor out) const override {
    if (!supports(projection, ape, sources, slots, state, out))
      throw std::invalid_argument("DeepSeek V4 compressor inputs changed");
    using namespace v4_compress_detail;
    const auto grid = [](size_t rows) {
      return static_cast<unsigned>(rows < 65535 ? rows : 65535);
    };
    auto stream = static_cast<cudaStream_t>(stream_);
    auto* state_data = static_cast<float*>(state.data());
    const auto launch = [&](auto tag) {
      using T = decltype(tag);
      const auto* data = static_cast<const T*>(projection.data());
      Pool<T><<<grid(out.size(0)), 128, 0, stream>>>(
          data, static_cast<const float*>(ape.data()),
          static_cast<const int32_t*>(sources.data()), state_data,
          static_cast<T*>(out.data()), projection.size(0), state.size(0),
          out.size(0), out.size(1), sources.size(1), ape.size(0) == 4);
      if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("DeepSeek V4 pooling launch failed");
      Update<T><<<grid(projection.size(0)), 128, 0, stream>>>(
          data, static_cast<const int32_t*>(slots.data()), state_data, projection.size(0),
          state.size(0), projection.size(1));
      if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("DeepSeek V4 state update launch failed");
    };
    if (projection.dtype() == DataType::kFloat32) launch(float{});
    else launch(__nv_bfloat16{});
  }
};

}  // namespace infini::ops

#endif
