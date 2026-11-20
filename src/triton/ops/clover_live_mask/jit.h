#pragma once
#include "base/clover_live_mask.h"
#include "triton/jit/jit.h"

namespace infini::ops::triton::jit {

template <Device::Type kDev>
class CloverLiveMask : public ::infini::ops::CloverLiveMask {
 public:
  using ::infini::ops::CloverLiveMask::CloverLiveMask;

  void operator()(const Tensor ids, Tensor previous, int64_t mask_id,
                  bool refresh, Tensor live) const override {
    if (ids.numel() == 0) return;
    const Config config{4u,
                        1u,
                        {{"N", ids.numel()},
                         {"WIDTH", 1},
                         {"MASK", mask_id},
                         {"REFRESH", refresh},
                         {"STAGE", 0},
                         {"BLOCK", 256}}};
    Launch<kDev>(
        "clover_live_rows", ids.device().index(), this->stream_,
        this->config_ptr_.get(), config, {},
        [&](const Config&) -> std::optional<Grid> {
          return Grid{
              static_cast<unsigned>((ids.numel() + 255) / 256)};
        },
        ids, previous, live, live);
  }
};

}  // namespace infini::ops::triton::jit

namespace infini::ops {

template <>
class Operator<CloverLiveMask, Device::Type::kNvidia, 10>
    : public triton::jit::CloverLiveMask<Device::Type::kNvidia> {
 public:
  using triton::jit::CloverLiveMask<Device::Type::kNvidia>::CloverLiveMask;
};

template <>
class Operator<CloverLiveMask, Device::Type::kHygon, 10>
    : public triton::jit::CloverLiveMask<Device::Type::kHygon> {
 public:
  using triton::jit::CloverLiveMask<Device::Type::kHygon>::CloverLiveMask;
};

}  // namespace infini::ops
