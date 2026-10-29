#pragma once
#include "base/clover_mask_routes.h"
#include "triton/jit/jit.h"

namespace infini::ops {

template <>
class Operator<CloverMaskRoutes, Device::Type::kHygon, 10>
    : public CloverMaskRoutes {
 public:
  using CloverMaskRoutes::CloverMaskRoutes;

  void operator()(const Tensor ids, const Tensor live, int64_t num_experts,
                  Tensor out) const override {
    if (ids.numel() == 0) return;
    const triton::jit::Config config{4u,
                                     1u,
                                     {{"N", ids.numel()},
                                      {"WIDTH", ids.size(1)},
                                      {"MASK", num_experts},
                                      {"REFRESH", false},
                                      {"STAGE", 1},
                                      {"BLOCK", 256}}};
    triton::jit::Launch<Device::Type::kHygon>(
        "clover_live_rows", ids.device().index(), stream_, nullptr, config, {},
        [&](const triton::jit::Config&) -> std::optional<triton::jit::Grid> {
          return triton::jit::Grid{
              static_cast<unsigned>((ids.numel() + 255) / 256)};
        },
        ids, live, out, out);
  }
};

}  // namespace infini::ops
