#pragma once
#include "base/clover_merge_moe.h"
#include "triton/jit/jit.h"

namespace infini::ops::triton::jit {

template <Device::Type kDev>
class CloverMergeMoe : public ::infini::ops::CloverMergeMoe {
 public:
  using ::infini::ops::CloverMergeMoe::CloverMergeMoe;

  void operator()(const Tensor fresh, const Tensor live, Tensor cache,
                  Tensor out) const override {
    if (fresh.numel() == 0) return;
    const Config config{4u,
                        1u,
                        {{"N", fresh.numel()},
                         {"WIDTH", fresh.size(1)},
                         {"MASK", 0},
                         {"REFRESH", false},
                         {"STAGE", 2},
                         {"BLOCK", 256}}};
    Launch<kDev>(
        "clover_live_rows", fresh.device().index(), this->stream_,
        this->config_ptr_.get(), config, {},
        [&](const Config&) -> std::optional<Grid> {
          return Grid{static_cast<unsigned>((fresh.numel() + 255) / 256)};
        },
        fresh, live, cache, out);
  }
};

}  // namespace infini::ops::triton::jit

namespace infini::ops {

template <>
class Operator<CloverMergeMoe, Device::Type::kNvidia, 10>
    : public triton::jit::CloverMergeMoe<Device::Type::kNvidia> {
 public:
  using triton::jit::CloverMergeMoe<Device::Type::kNvidia>::CloverMergeMoe;
};

template <>
class Operator<CloverMergeMoe, Device::Type::kHygon, 10>
    : public triton::jit::CloverMergeMoe<Device::Type::kHygon> {
 public:
  using triton::jit::CloverMergeMoe<Device::Type::kHygon>::CloverMergeMoe;
};

}  // namespace infini::ops
