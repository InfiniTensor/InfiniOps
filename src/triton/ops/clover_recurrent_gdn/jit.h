#pragma once

#include "base/clover_recurrent_gdn.h"
#include "triton/jit/jit.h"

namespace infini::ops::triton::jit {

template <Device::Type kDev>
class CloverRecurrentGdn : public ::infini::ops::CloverRecurrentGdn {
 public:
  using ::infini::ops::CloverRecurrentGdn::CloverRecurrentGdn;

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor g, const Tensor beta, const Tensor offsets,
                  const Tensor initial_indices, const Tensor final_indices,
                  Tensor state, Tensor out) const override {
    const Config config{4u,
                        1u,
                        {{"TOKENS", q.size(1)},
                         {"POOL", state.size(0)},
                         {"HK", q.size(2)},
                         {"HV", v.size(2)},
                         {"BV", 16}}};
    Launch<kDev>(
        "clover_recurrent_gdn", q.device().index(), this->stream_,
        this->config_ptr_.get(), config, {},
        [&](const Config&) -> std::optional<Grid> {
          return Grid{static_cast<unsigned>(offsets.numel() - 1),
                      static_cast<unsigned>(v.size(2)), 8};
        },
        q, k, v, g, beta, offsets, initial_indices, final_indices, state, out,
        q.stride(1), q.stride(2), k.stride(1), k.stride(2), v.stride(1),
        v.stride(2), g.stride(1), g.stride(2), beta.stride(1), beta.stride(2),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        out.stride(1), out.stride(2));
  }
};

}  // namespace infini::ops::triton::jit

namespace infini::ops {

template <>
class Operator<CloverRecurrentGdn, Device::Type::kHygon, 10>
    : public triton::jit::CloverRecurrentGdn<Device::Type::kHygon> {
 public:
  using triton::jit::CloverRecurrentGdn<
      Device::Type::kHygon>::CloverRecurrentGdn;
};

}  // namespace infini::ops
