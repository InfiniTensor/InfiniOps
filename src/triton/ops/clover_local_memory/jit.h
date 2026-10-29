#pragma once

#include <cmath>

#include "base/clover_local_memory.h"
#include "triton/jit/jit.h"

namespace infini::ops::triton::jit {

template <Device::Type kDev>
class CloverLocalMemory : public ::infini::ops::CloverLocalMemory {
 public:
  using ::infini::ops::CloverLocalMemory::CloverLocalMemory;

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor g, const Tensor state, const Tensor indices,
                  const Tensor valid, const Tensor logit_scale,
                  const Tensor history_bias, const Tensor history_c,
                  size_t block_size, Tensor out) const override {
    if (q.numel() == 0) return;
    const Config config{4u,
                        1u,
                        {{"NUM_Q_HEADS", q.size(2)},
                         {"NUM_K_HEADS", k.size(2)},
                         {"NUM_V_HEADS", v.size(2)},
                         {"BLOCK_SIZE", block_size},
                         {"NUM_SLOTS", state.size(0)},
                         {"KEY_DIM", 128},
                         {"VALUE_DIM", 128},
                         {"BLOCK_T", 32},
                         {"BLOCK_Q", 32},
                         {"NUM_QUERY_TILES", 1},
                         {"BLOCK_K", 128},
                         {"BLOCK_V", 64}}};
    Launch<kDev>(
        "clover_local_memory", q.device().index(), this->stream_,
        this->config_ptr_.get(), config, {},
        [&](const Config&) -> std::optional<Grid> {
          return Grid{2, static_cast<unsigned>(indices.numel()),
                      static_cast<unsigned>(v.size(2))};
        },
        q, k, v, g, state, indices, valid, logit_scale, history_bias, history_c,
        out, q.stride(1), q.stride(2), q.stride(3), k.stride(1), k.stride(2),
        k.stride(3), v.stride(1), v.stride(2), v.stride(3), g.stride(1),
        g.stride(2), state.stride(0), state.stride(1), state.stride(2),
        state.stride(3), out.stride(1), out.stride(2), out.stride(3),
        1.0f / std::sqrt(128.0f), 32.0f);
  }
};

}  // namespace infini::ops::triton::jit

namespace infini::ops {

template <>
class Operator<CloverLocalMemory, Device::Type::kHygon, 10>
    : public triton::jit::CloverLocalMemory<Device::Type::kHygon> {
 public:
  using triton::jit::CloverLocalMemory<Device::Type::kHygon>::CloverLocalMemory;
};

}  // namespace infini::ops
