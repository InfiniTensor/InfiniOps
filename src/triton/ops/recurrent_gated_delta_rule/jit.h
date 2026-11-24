#pragma once

#include <stdexcept>

#include "base/recurrent_gated_delta_rule.h"
#include "triton/jit/jit.h"

namespace infini::ops::triton::jit {

// Triton reference for the packed form of `RecurrentGatedDeltaRule`.
template <Device::Type kDev>
class PackedRecurrentGatedDeltaRule
    : public ::infini::ops::RecurrentGatedDeltaRule {
 public:
  using ::infini::ops::RecurrentGatedDeltaRule::RecurrentGatedDeltaRule;

  // The batched single-token decode is left to the native providers.
  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor g, const Tensor beta, Tensor initial_state,
                  const std::optional<Tensor> initial_state_indices,
                  const std::optional<Tensor> final_state_indices, Tensor out,
                  std::optional<Tensor> final_state,
                  bool use_qk_l2norm) const override {
    throw std::invalid_argument(
        "the Triton RecurrentGatedDeltaRule provider implements the packed "
        "form only");
  }

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor g, const Tensor beta, Tensor initial_state,
                  const Tensor cu_seqlens, const Tensor initial_slots,
                  const Tensor final_slots, Tensor out,
                  bool use_qk_l2norm) const override {
    const Config config{4u,
                        1u,
                        {{"TOKENS", q.size(1)},
                         {"POOL", initial_state.size(0)},
                         {"HK", q.size(2)},
                         {"HV", v.size(2)},
                         {"BV", 16},
                         {"NORM", use_qk_l2norm}}};
    Launch<kDev>(
        "recurrent_gated_delta_rule", q.device().index(), this->stream_,
        this->config_ptr_.get(), config, {},
        [&](const Config&) -> std::optional<Grid> {
          return Grid{static_cast<unsigned>(cu_seqlens.numel() - 1),
                      static_cast<unsigned>(v.size(2)), 8};
        },
        q, k, v, g, beta, cu_seqlens, initial_slots,
        final_slots, initial_state, out, q.stride(1), q.stride(2),
        k.stride(1), k.stride(2), v.stride(1), v.stride(2), g.stride(1),
        g.stride(2), beta.stride(1), beta.stride(2), initial_state.stride(0),
        initial_state.stride(1), initial_state.stride(2),
        initial_state.stride(3), out.stride(1), out.stride(2));
  }
};

}  // namespace infini::ops::triton::jit

namespace infini::ops {

template <>
class Operator<RecurrentGatedDeltaRule, Device::Type::kNvidia, 10>
    : public triton::jit::PackedRecurrentGatedDeltaRule<Device::Type::kNvidia> {
 public:
  using triton::jit::PackedRecurrentGatedDeltaRule<
      Device::Type::kNvidia>::PackedRecurrentGatedDeltaRule;
};

template <>
class Operator<RecurrentGatedDeltaRule, Device::Type::kHygon, 10>
    : public triton::jit::PackedRecurrentGatedDeltaRule<Device::Type::kHygon> {
 public:
  using triton::jit::PackedRecurrentGatedDeltaRule<
      Device::Type::kHygon>::PackedRecurrentGatedDeltaRule;
};

}  // namespace infini::ops
