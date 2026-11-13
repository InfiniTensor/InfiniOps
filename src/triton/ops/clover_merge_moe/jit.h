#pragma once
#include "base/clover_merge_moe.h"
#include "triton/jit/jit.h"

namespace infini::ops {

template <>
class Operator<CloverMergeMoe, Device::Type::kHygon, 10>
    : public CloverMergeMoe {
 public:
  using CloverMergeMoe::CloverMergeMoe;

  void operator()(const Tensor fresh, const Tensor live, Tensor cache,
                  Tensor out) const override {
    if (fresh.numel() == 0) return;
    const triton::jit::Config config{4u,
                                     1u,
                                     {{"N", fresh.numel()},
                                      {"WIDTH", fresh.size(1)},
                                      {"MASK", 0},
                                      {"REFRESH", false},
                                      {"STAGE", 2},
                                      {"BLOCK", 256}}};
    triton::jit::Launch<Device::Type::kHygon>(
        "clover_live_rows", fresh.device().index(), stream_, nullptr, config,
        {},
        [&](const triton::jit::Config&) -> std::optional<triton::jit::Grid> {
          return triton::jit::Grid{
              static_cast<unsigned>((fresh.numel() + 255) / 256)};
        },
        fresh, live, cache, out);
  }
};

}  // namespace infini::ops
