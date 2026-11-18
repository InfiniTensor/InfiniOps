#ifndef INFINI_OPS_KUNLUN_CHUNK_GATED_DELTA_RULE_KERNEL_H_
#define INFINI_OPS_KUNLUN_CHUNK_GATED_DELTA_RULE_KERNEL_H_

#include <stdexcept>

#include "base/chunk_gated_delta_rule.h"

namespace infini::ops {

// Placeholder:  has no Kunlun implementation yet. The class must
// exist so downstream consumers (InfiniLM) referencing the operator type
// at compile time build; running it throws.
template <>
class Operator<ChunkGatedDeltaRule, Device::Type::kKunlun>
    : public ChunkGatedDeltaRule {
 public:
  using ChunkGatedDeltaRule::ChunkGatedDeltaRule;

  [[noreturn]] void operator()(
      const Tensor q, const Tensor k, const Tensor v, const Tensor g,
      const Tensor beta, Tensor initial_state,
      const std::optional<Tensor> cu_seqlens,
      const std::optional<Tensor> initial_state_indices,
      const std::optional<Tensor> final_state_indices, Tensor out,
      std::optional<Tensor> final_state, bool use_qk_l2norm,
      size_t chunk_size) const override {
    throw std::runtime_error(
        "ChunkGatedDeltaRule is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
