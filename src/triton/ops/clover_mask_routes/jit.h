#pragma once
#include "base/clover_mask_routes.h"
#include "triton/jit/jit.h"

namespace infini::ops::triton::jit {

template <Device::Type kDev>
class CloverMaskRoutes : public ::infini::ops::CloverMaskRoutes {
 public:
  using ::infini::ops::CloverMaskRoutes::CloverMaskRoutes;

  void operator()(const Tensor ids, const Tensor live, int64_t num_experts,
                  Tensor out) const override {
    if (ids.numel() == 0) return;
    const Config config{4u,
                        1u,
                        {{"N", ids.numel()},
                         {"WIDTH", ids.size(1)},
                         {"MASK", num_experts},
                         {"REFRESH", false},
                         {"STAGE", 1},
                         {"BLOCK", 256}}};
    Launch<kDev>(
        "clover_live_rows", ids.device().index(), this->stream_,
        this->config_ptr_.get(), config, {},
        [&](const Config&) -> std::optional<Grid> {
          return Grid{
              static_cast<unsigned>((ids.numel() + 255) / 256)};
        },
        ids, live, out, out);
  }
};

}  // namespace infini::ops::triton::jit

namespace infini::ops {

template <>
class Operator<CloverMaskRoutes, Device::Type::kNvidia, 10>
    : public triton::jit::CloverMaskRoutes<Device::Type::kNvidia> {
 public:
  using triton::jit::CloverMaskRoutes<Device::Type::kNvidia>::CloverMaskRoutes;
};

template <>
class Operator<CloverMaskRoutes, Device::Type::kHygon, 10>
    : public triton::jit::CloverMaskRoutes<Device::Type::kHygon> {
 public:
  using triton::jit::CloverMaskRoutes<Device::Type::kHygon>::CloverMaskRoutes;
};

}  // namespace infini::ops
