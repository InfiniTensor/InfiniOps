#pragma once

#include "base/moe_fused_dense.h"
#include "native/cuda/device_buffer.h"
#include "triton/jit/jit.h"

namespace infini::ops::triton::jit {

template <Device::Type kDev>
class MoeFusedDense : public ::infini::ops::MoeFusedDense {
 public:
  using ::infini::ops::MoeFusedDense::MoeFusedDense;

  static bool supports(Tensor output, const Tensor hidden, const Tensor w13,
                       const Tensor w2, const Tensor weights, const Tensor ids,
                       const Tensor sorted, const Tensor experts,
                       const Tensor total, size_t block_size,
                       const std::optional<Tensor> bias1 = std::nullopt,
                       const std::optional<Tensor> bias2 = std::nullopt) {
    return block_size == 16 && !bias1 && !bias2 &&
           ::infini::ops::MoeFusedDense::supports(
               output, hidden, w13, w2, weights, ids, sorted, experts, total,
               block_size, bias1, bias2);
  }

  size_t workspace_size_in_bytes() const override {
    const auto pairs = metadata_[1].size(0) * metadata_[5].size(1);
    return pairs * (metadata_[3].size(2) * 2 + metadata_[1].size(1)) *
           metadata_[1].element_size();
  }

  void operator()(Tensor output, const Tensor hidden, const Tensor w13,
                  const Tensor w2, const Tensor weights, const Tensor ids,
                  const Tensor sorted, const Tensor experts, const Tensor total,
                  size_t block_size, const std::optional<Tensor> bias1,
                  const std::optional<Tensor> bias2) const override {
    Invocation(output, hidden, w13, w2, weights, ids, sorted, experts, total,
               block_size, bias1, bias2);
    if (bias1 || bias2)
      throw std::invalid_argument(
          "Hygon Triton MoE does not support expert biases.");
    // The Python bindings call without a handle: use the caller's workspace
    // when it is large enough, else a buffer owned by this operator.
    void* scratch =
        workspace_ && workspace_size_in_bytes_ >= workspace_size_in_bytes()
            ? workspace_
            : fallback_.Allocate(workspace_size_in_bytes());
    const auto n = hidden.size(0), h = hidden.size(1), i = w2.size(2),
               topk = ids.size(1);
    const auto pairs = n * topk;
    Tensor mid{scratch, {pairs, 2 * i}, hidden.dtype(), hidden.device()};
    Tensor pair_output{
        static_cast<char*>(scratch) + pairs * 2 * i * hidden.element_size(),
        {pairs, h},
        hidden.dtype(),
        hidden.device()};
    for (int stage = 0; stage < 3; ++stage) {
      const Config config{4u,
                          2u,
                          {{"TOKENS", n},
                           {"E", w13.size(0)},
                           {"H", h},
                           {"I", i},
                           {"TOPK", topk},
                           {"BM", block_size},
                           {"BN", 64},
                           {"BK", 64},
                           {"STAGE", stage}}};
      Launch<kDev>(
          "moe_fused_dense", hidden.device().index(), this->stream_, nullptr,
          config, {},
          [&](const Config&) -> std::optional<Grid> {
            if (stage == 2)
              return Grid{static_cast<unsigned>(n),
                          static_cast<unsigned>((h + 63) / 64)};
            const auto cols = stage == 0 ? 2 * i : h;
            return Grid{static_cast<unsigned>(sorted.numel() / block_size),
                        static_cast<unsigned>((cols + 63) / 64)};
          },
          hidden, w13, w2, weights, ids, sorted, experts, total, mid,
          pair_output, output);
    }
  }

 private:
  mutable CudaDeviceBuffer<Runtime<kDev>> fallback_;
};

}  // namespace infini::ops::triton::jit

namespace infini::ops {

template <>
class Operator<MoeFusedDense, Device::Type::kNvidia, 10>
    : public triton::jit::MoeFusedDense<Device::Type::kNvidia> {
 public:
  using triton::jit::MoeFusedDense<Device::Type::kNvidia>::MoeFusedDense;
};

template <>
class Operator<MoeFusedDense, Device::Type::kHygon, 10>
    : public triton::jit::MoeFusedDense<Device::Type::kHygon> {
 public:
  using triton::jit::MoeFusedDense<Device::Type::kHygon>::MoeFusedDense;
};

}  // namespace infini::ops
