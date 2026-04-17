#ifndef INFINI_OPS_BASE_RANDOM_SAMPLE_H_
#define INFINI_OPS_BASE_RANDOM_SAMPLE_H_

#include <cstdint>
#include <optional>

#include "operator.h"

namespace infini::ops {

class RandomSample : public Operator<RandomSample> {
 public:
  // clang-format off
  //
  // logits:  [batch_size, vocab_size] or [vocab_size] (batch_size=1)
  // out:     [batch_size]  sampled token ids (int32/int64)
  // valid:   [batch_size]  bool, whether sample is valid
  //
  // Per-batch parameters support two modes:
  //   - optional<Tensor> has value: per-batch tensor of shape [batch_size]
  //   - optional<Tensor> is nullopt: use the scalar _val for all requests
  //
  // When both (optional<Tensor> == nullopt) and (_val == default),
  // the corresponding filtering is disabled.
  //
  // clang-format on
  RandomSample(const Tensor logits, Tensor out, Tensor valid,
               std::optional<Tensor> temperature, float temperature_val,
               std::optional<Tensor> top_k, int top_k_val,
               std::optional<Tensor> top_p, float top_p_val,
               std::optional<Tensor> min_p, float min_p_val,
               std::uint64_t seed, std::uint64_t offset,
               bool deterministic)
      : logits_dtype_{logits.dtype()},
        out_dtype_{out.dtype()},
        ndim_{logits.ndim()},
        batch_size_{ndim_ == 2 ? logits.size(-2) : 1},
        vocab_size_{logits.size(-1)},
        logits_strides_{logits.strides()},
        temperature_{temperature},
        temperature_val_{temperature_val},
        top_k_{top_k},
        top_k_val_{top_k_val},
        top_p_{top_p},
        top_p_val_{top_p_val},
        min_p_{min_p},
        min_p_val_{min_p_val},
        seed_{seed},
        offset_{offset},
        deterministic_{deterministic} {
    assert((ndim_ == 1 || ndim_ == 2) &&
           "`RandomSample` requires 1D [vocab_size] or 2D [batch, vocab_size] "
           "logits");
    assert(out.ndim() == 1 && out.size(0) == batch_size_ &&
           "`RandomSample` requires 1D output [batch_size]");
    assert(valid.ndim() == 1 && valid.size(0) == batch_size_ &&
           "`RandomSample` requires 1D valid [batch_size]");
  }

  // Simplified constructor: no filtering, default temperature.
  RandomSample(const Tensor logits, Tensor out, Tensor valid,
               std::uint64_t seed, std::uint64_t offset)
      : RandomSample{logits, out, valid,
                     std::nullopt, 1.0f,
                     std::nullopt, 0,
                     std::nullopt, 1.0f,
                     std::nullopt, 0.0f,
                     seed, offset, false} {}

  virtual void operator()(const Tensor logits, Tensor out, Tensor valid,
                          std::optional<Tensor> temperature,
                          float temperature_val,
                          std::optional<Tensor> top_k, int top_k_val,
                          std::optional<Tensor> top_p, float top_p_val,
                          std::optional<Tensor> min_p, float min_p_val,
                          std::uint64_t seed, std::uint64_t offset,
                          bool deterministic) const = 0;

  virtual void operator()(const Tensor logits, Tensor out, Tensor valid,
                          std::uint64_t seed, std::uint64_t offset) const {
    return operator()(logits, out, valid,
                      std::nullopt, temperature_val_,
                      std::nullopt, top_k_val_,
                      std::nullopt, top_p_val_,
                      std::nullopt, min_p_val_,
                      seed, offset, deterministic_);
  }

 protected:
  const DataType logits_dtype_;

  const DataType out_dtype_;

  Tensor::Size ndim_{0};

  Tensor::Size batch_size_{1};

  Tensor::Size vocab_size_{0};

  Tensor::Strides logits_strides_;

  // Per-batch or scalar sampling parameters.
  std::optional<Tensor> temperature_;
  float temperature_val_{1.0f};

  std::optional<Tensor> top_k_;
  int top_k_val_{0};

  std::optional<Tensor> top_p_;
  float top_p_val_{1.0f};

  std::optional<Tensor> min_p_;
  float min_p_val_{0.0f};

  std::uint64_t seed_{0};

  std::uint64_t offset_{0};

  bool deterministic_{false};
};

}  // namespace infini::ops

#endif
