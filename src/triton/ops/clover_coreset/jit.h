#pragma once

#include "base/clover_coreset.h"
#include "triton/jit/jit.h"

namespace infini::ops {

template <>
class Operator<CloverCoreset, Device::Type::kHygon, 10> : public CloverCoreset {
 public:
  Operator(const Tensor logits, size_t block_size, double top_p, Tensor out)
      : CloverCoreset(logits, block_size, top_p, out),
        blocks_(logits.size(0) / block_size) {}

  size_t workspace_size_in_bytes() const override { return blocks_ * 256 * 5; }

  void operator()(const Tensor logits, size_t block_size, double top_p,
                  Tensor out) const override {
    if (!blocks_) return;
    if (!workspace_ || workspace_size_in_bytes_ < workspace_size_in_bytes())
      throw std::invalid_argument("CloverCoreset: missing workspace.");
    Tensor mass{
        workspace_, {blocks_, 256}, DataType::kFloat32, logits.device()};
    Tensor allow{static_cast<char*>(workspace_) + blocks_ * 256 * 4,
                 {blocks_, 256},
                 DataType::kInt8,
                 logits.device()};
    for (int stage = 0; stage < 3; ++stage) {
      const triton::jit::Config config{
          4u, 1u, {{"WIDTH", block_size}, {"STAGE", stage}}};
      triton::jit::Launch<Device::Type::kHygon>(
          "clover_coreset", logits.device().index(), stream_, nullptr, config,
          {},
          [&](const triton::jit::Config&) -> std::optional<triton::jit::Grid> {
            return triton::jit::Grid{
                static_cast<unsigned>(stage == 2 ? logits.size(0) : blocks_)};
          },
          logits, mass, allow, out, static_cast<float>(top_p));
    }
  }

 private:
  size_t blocks_;
};

}  // namespace infini::ops
