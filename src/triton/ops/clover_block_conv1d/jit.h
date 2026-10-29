#pragma once

#include "base/clover_block_conv1d.h"
#include "triton/jit/jit.h"

namespace infini::ops::triton::jit {

template <Device::Type kDev>
class CloverBlockConv1d : public ::infini::ops::CloverBlockConv1d {
 public:
  using ::infini::ops::CloverBlockConv1d::CloverBlockConv1d;

  void operator()(const Tensor input, const Tensor state, const Tensor indices,
                  const Tensor weight, const Tensor right_weight,
                  const Tensor valid, size_t block_size,
                  Tensor out) const override {
    if (input.numel() == 0) return;
    const Config config{4u,
                        1u,
                        {{"NUM_CHANNELS", input.size(1)},
                         {"BLOCK_SIZE", block_size},
                         {"NUM_SLOTS", state.size(0)},
                         {"WIDTH", 4},
                         {"HAS_BIAS", 0},
                         {"INDEPENDENT_SYMMETRIC", 1},
                         {"BLOCK_T", 4},
                         {"NUM_TOKEN_TILES", (block_size + 3) / 4},
                         {"BLOCK_C", 128}}};
    Launch<kDev>(
        "clover_block_conv1d", input.device().index(), this->stream_,
        this->config_ptr_.get(), config, {},
        [&](const Config&) -> std::optional<Grid> {
          return Grid{
              static_cast<unsigned>((input.size(1) + 127) / 128),
              static_cast<unsigned>(indices.numel() * ((block_size + 3) / 4))};
        },
        input, state, indices, weight, weight, right_weight, valid, out,
        input.stride(0), input.stride(1), state.stride(0), state.stride(1),
        state.stride(2), weight.stride(0), weight.stride(2),
        right_weight.stride(0), right_weight.stride(2), out.stride(0),
        out.stride(1));
  }
};

}  // namespace infini::ops::triton::jit

namespace infini::ops {

template <>
class Operator<CloverBlockConv1d, Device::Type::kHygon, 10>
    : public triton::jit::CloverBlockConv1d<Device::Type::kHygon> {
 public:
  using triton::jit::CloverBlockConv1d<Device::Type::kHygon>::CloverBlockConv1d;
};

}  // namespace infini::ops
