#pragma once
#include "base/top_k_top_p_sampling_from_logits.h"

#define HYGON_SAMPLING_ARGS                                           \
  const Tensor logits, const Tensor top_k, const Tensor top_p,        \
      const std::optional<Tensor> indices,                            \
      const std::string filter_apply_order, const bool deterministic, \
      const bool check_nan, const std::optional<int64_t> seed,        \
      const std::optional<int64_t> offset, Tensor out
#define HYGON_SAMPLING_VALUES                                                  \
  logits, top_k, top_p, indices, filter_apply_order, deterministic, check_nan, \
      seed, offset, out

namespace infini::ops {
template <>
class Operator<TopKTopPSamplingFromLogits, Device::Type::kHygon, 8>
    : public TopKTopPSamplingFromLogits {
 public:
  static bool supports(HYGON_SAMPLING_ARGS);
  Operator(HYGON_SAMPLING_ARGS);
  void operator()(HYGON_SAMPLING_ARGS) const override;

 private:
  static Tensor Checked(HYGON_SAMPLING_ARGS) {
    if (!supports(HYGON_SAMPLING_VALUES))
      throw std::invalid_argument("unsupported Hygon sampling metadata");
    return logits;
  }
};
}  // namespace infini::ops
