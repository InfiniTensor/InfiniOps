#ifndef INFINI_OPS_CPU_RANDOM_SAMPLE_H_
#define INFINI_OPS_CPU_RANDOM_SAMPLE_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include "base/random_sample.h"
#include "common/generic_utils.h"
#include "cpu/caster_.h"
#include "data_type.h"
#include "tensor.h"

namespace infini::ops {

template <>
class Operator<RandomSample, Device::Type::kCpu>
    : public RandomSample,
      Caster<Device::Type::kCpu> {
 public:
  using RandomSample::RandomSample;

  void operator()(const Tensor logits, Tensor out, Tensor valid,
                  std::optional<Tensor> temperature, float temperature_val,
                  std::optional<Tensor> top_k, int top_k_val,
                  std::optional<Tensor> top_p, float top_p_val,
                  std::optional<Tensor> min_p, float min_p_val,
                  std::uint64_t seed, std::uint64_t offset,
                  bool deterministic) const override {
    DispatchFunc<ConcatType<FloatTypes, ReducedFloatTypes>>(
        logits.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(logits, out, valid, temperature, temperature_val, top_k,
                     top_k_val, top_p, top_p_val, min_p, min_p_val, seed,
                     offset);
        },
        "`Operator<RandomSample, Device::Type::kCpu>::operator()`");
  }

  void operator()(const Tensor logits, Tensor out, Tensor valid,
                  std::uint64_t seed, std::uint64_t offset) const override {
    return operator()(logits, out, valid,
                      std::nullopt, temperature_val_,
                      std::nullopt, top_k_val_,
                      std::nullopt, top_p_val_,
                      std::nullopt, min_p_val_,
                      seed, offset);
  }

 private:
  // Resolve a per-batch parameter: use tensor if provided, else scalar.
  template <typename ValType>
  ValType GetParam(std::optional<Tensor> tensor, ValType scalar_val,
                   Tensor::Size batch_idx) const {
    if (tensor.has_value()) {
      const auto* ptr = static_cast<const ValType*>(tensor->data());
      return ptr[batch_idx];
    }
    return scalar_val;
  }

  // Specialization for float temperature from a typed tensor.
  float GetTemperature(std::optional<Tensor> tensor, float scalar_val,
                       Tensor::Size batch_idx) const {
    if (tensor.has_value()) {
      // Temperature tensor may be float16/bfloat16/float32.
      const auto& t = tensor.value();
      if (t.dtype() == DataType::kFloat32) {
        return static_cast<const float*>(t.data())[batch_idx];
      } else if (t.dtype() == DataType::kFloat16) {
        auto v = static_cast<const Float16*>(t.data())[batch_idx];
        return v.ToFloat();
      } else if (t.dtype() == DataType::kBFloat16) {
        auto v = static_cast<const BFloat16*>(t.data())[batch_idx];
        return v.ToFloat();
      }
    }
    return scalar_val;
  }

  template <typename T>
  void Compute(const Tensor logits, Tensor out, Tensor valid,
               std::optional<Tensor> temperature, float temperature_val,
               std::optional<Tensor> top_k, int top_k_val,
               std::optional<Tensor> top_p, float top_p_val,
               std::optional<Tensor> min_p, float min_p_val,
               std::uint64_t seed, std::uint64_t offset) const {
    const auto* logits_ptr = static_cast<const T*>(logits.data());
    auto* out_ptr = static_cast<std::int32_t*>(out.data());
    auto* valid_ptr = static_cast<bool*>(valid.data());

    auto vocab_size = static_cast<Tensor::Size>(vocab_size_);

    // Working buffer for one row of float probabilities.
    std::vector<float> probs(vocab_size);

#pragma omp parallel for firstprivate(probs)
    for (Tensor::Size b = 0; b < batch_size_; ++b) {
      const T* logits_row = logits_ptr + b * vocab_size;

      // --- Step 1: Temperature scaling + Softmax ---
      float temp = GetTemperature(temperature, temperature_val, b);
      float inv_temp = (temp > 0.f) ? (1.f / temp) : 0.f;

      float max_val = Cast<float>(logits_row[0]) * inv_temp;
      for (Tensor::Size j = 1; j < vocab_size; ++j) {
        float v = Cast<float>(logits_row[j]) * inv_temp;
        if (v > max_val) max_val = v;
      }

      float sum = 0.f;
      for (Tensor::Size j = 0; j < vocab_size; ++j) {
        float v = std::exp(Cast<float>(logits_row[j]) * inv_temp - max_val);
        probs[j] = v;
        sum += v;
      }

      if (sum <= 0.f) {
        out_ptr[b] = 0;
        valid_ptr[b] = false;
        continue;
      }

      for (Tensor::Size j = 0; j < vocab_size; ++j) {
        probs[j] /= sum;
      }

      // --- Step 2: top_k filtering ---
      int k = GetParam<int>(top_k, top_k_val, b);
      if (k > 0 && static_cast<Tensor::Size>(k) < vocab_size) {
        // Find the k-th largest value using nth_element.
        std::vector<std::pair<float, Tensor::Size>> indexed(vocab_size);
        for (Tensor::Size j = 0; j < vocab_size; ++j) {
          indexed[j] = {probs[j], j};
        }
        // Partial sort: top k elements at the front.
        std::nth_element(
            indexed.begin(), indexed.begin() + k, indexed.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        // Mask out everything beyond top-k.
        for (Tensor::Size j = static_cast<Tensor::Size>(k); j < vocab_size;
             ++j) {
          probs[indexed[j].second] = 0.f;
        }

        // Renormalize.
        float renorm_sum = 0.f;
        for (Tensor::Size j = 0; j < vocab_size; ++j) {
          renorm_sum += probs[j];
        }
        if (renorm_sum > 0.f) {
          for (Tensor::Size j = 0; j < vocab_size; ++j) {
            probs[j] /= renorm_sum;
          }
        }
      }

      // --- Step 3: top_p filtering ---
      float p = GetParam<float>(top_p, top_p_val, b);
      if (p > 0.f && p < 1.f) {
        // Sort indices by probability descending.
        std::vector<Tensor::Size> sorted_idx(vocab_size);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(),
                  [&](Tensor::Size a, Tensor::Size b) {
                    return probs[a] > probs[b];
                  });

        float cumsum = 0.f;
        Tensor::Size cutoff = vocab_size;
        for (Tensor::Size i = 0; i < vocab_size; ++i) {
          cumsum += probs[sorted_idx[i]];
          if (cumsum >= p) {
            cutoff = i + 1;
            break;
          }
        }
        // Zero out everything beyond the cutoff.
        for (Tensor::Size i = cutoff; i < vocab_size; ++i) {
          probs[sorted_idx[i]] = 0.f;
        }

        // Renormalize.
        float renorm_sum = 0.f;
        for (Tensor::Size j = 0; j < vocab_size; ++j) {
          renorm_sum += probs[j];
        }
        if (renorm_sum > 0.f) {
          for (Tensor::Size j = 0; j < vocab_size; ++j) {
            probs[j] /= renorm_sum;
          }
        }
      }

      // --- Step 4: min_p filtering ---
      float mp = GetParam<float>(min_p, min_p_val, b);
      if (mp > 0.f) {
        // Find max probability.
        float max_prob = *std::max_element(probs.begin(), probs.end());
        float threshold = max_prob * mp;

        for (Tensor::Size j = 0; j < vocab_size; ++j) {
          if (probs[j] < threshold) {
            probs[j] = 0.f;
          }
        }

        // Renormalize.
        float renorm_sum = 0.f;
        for (Tensor::Size j = 0; j < vocab_size; ++j) {
          renorm_sum += probs[j];
        }
        if (renorm_sum > 0.f) {
          for (Tensor::Size j = 0; j < vocab_size; ++j) {
            probs[j] /= renorm_sum;
          }
        }
      }

      // --- Step 5: Sample from CDF ---
      std::mt19937 rng(static_cast<std::uint64_t>(seed) +
                       static_cast<std::uint64_t>(b) + offset);
      std::uniform_real_distribution<float> dist(0.f, 1.f);
      float u = dist(rng);

      float cdf = 0.f;
      Tensor::Size sampled = 0;
      bool found = false;
      for (Tensor::Size j = 0; j < vocab_size; ++j) {
        cdf += probs[j];
        if (cdf > u) {
          sampled = j;
          found = true;
          break;
        }
      }

      if (!found) {
        // Fallback: pick the last non-zero probability token.
        for (Tensor::Size j = vocab_size; j > 0; --j) {
          if (probs[j - 1] > 0.f) {
            sampled = j - 1;
            found = true;
            break;
          }
        }
      }

      out_ptr[b] = static_cast<std::int32_t>(sampled);
      valid_ptr[b] = found;
    }
  }
};

}  // namespace infini::ops

#endif
