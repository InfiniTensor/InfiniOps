#pragma once
#include "base/clover_live_mask.h"
#include "triton/jit/jit.h"

namespace infini::ops {

template <>
class Operator<CloverLiveMask, Device::Type::kHygon, 10>
    : public CloverLiveMask {
 public:
  using CloverLiveMask::CloverLiveMask;

  void operator()(const Tensor ids, Tensor previous, int64_t mask_id,
                  bool refresh, Tensor live) const override {
    if (ids.numel() == 0) return;
    const triton::jit::Config config{4u,
                                     1u,
                                     {{"N", ids.numel()},
                                      {"WIDTH", 1},
                                      {"MASK", mask_id},
                                      {"REFRESH", refresh},
                                      {"STAGE", 0},
                                      {"BLOCK", 256}}};
    triton::jit::Launch<Device::Type::kHygon>(
        "clover_live_rows", ids.device().index(), stream_, nullptr, config, {},
        [&](const triton::jit::Config&) -> std::optional<triton::jit::Grid> {
          return triton::jit::Grid{
              static_cast<unsigned>((ids.numel() + 255) / 256)};
        },
        ids, previous, live, live);
  }
};

}  // namespace infini::ops
