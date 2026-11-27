#pragma once

#include "operator.h"

namespace infini::ops {

// Block attention of a Clover preview: every token of a block attends to the
// valid tokens of the same block and to one bucket that reads the clean state
// of its slot. `q`, `k` and `v` are `[1, tokens, heads, 128]` in BF16 or FP16,
// with the value heads grouped onto the query and key heads; `state` is the
// FP32 `[slots, value_heads, 128, 128]` pool, `g` the FP32 log decay per token
// and value head, `indices[r]` the slot of request r and `valid` the live
// rows. Query and key rows are L2-normalized. The history logit of a token is
// `history_bias + history_c * log(valid_tokens / block_size)` per value head,
// and its value is `exp(g) / sqrt(128)` times the state read. A row that is
// invalid, or that belongs to a request with an out-of-range slot, outputs
// zero. The state is only read.
class CloverLocalMemory : public Operator<CloverLocalMemory> {
 public:
  CloverLocalMemory(const Tensor q, const Tensor k, const Tensor v,
                    const Tensor g, const Tensor state, const Tensor indices,
                    const Tensor valid, const Tensor logit_scale,
                    const Tensor history_bias, const Tensor history_c,
                    size_t block_size, Tensor out) {
    assert((q.ndim() == 4 && q.size(0) == 1 && k.shape() == q.shape() &&
            v.ndim() == 4 && v.size(0) == 1 && v.size(1) == q.size(1) &&
            q.size(2) > 0 && v.size(2) > 0 && v.size(2) % q.size(2) == 0 &&
            q.size(3) == 128 && v.size(3) == 128 && out.shape() == v.shape() &&
            state.ndim() == 4 && state.size(1) == v.size(2) &&
            state.size(2) == 128 && state.size(3) == 128 &&
            indices.ndim() == 1 && block_size > 0 && block_size <= 32 &&
            q.size(1) == indices.numel() * block_size &&
            g.shape() == Tensor::Shape{1, q.size(1), v.size(2)} &&
            valid.shape() == Tensor::Shape{q.size(1)}) &&
           "CloverLocalMemory: invalid sequence, head or state shape.");
    assert(
        (q.dtype() == DataType::kBFloat16 || q.dtype() == DataType::kFloat16) &&
        k.dtype() == q.dtype() && v.dtype() == q.dtype() &&
        out.dtype() == q.dtype() && state.dtype() == DataType::kFloat32 &&
        g.dtype() == DataType::kFloat32 &&
        "CloverLocalMemory: expected reduced data and FP32 state/gates.");
    assert((indices.dtype() == DataType::kInt32 ||
            indices.dtype() == DataType::kInt64) &&
           indices.IsContiguous() && valid.dtype() == DataType::kInt8 &&
           valid.IsContiguous() &&
           "CloverLocalMemory: invalid index/validity dtype.");
    for (const auto& scalar : {logit_scale, history_bias, history_c}) {
      assert(scalar.ndim() == 1 && scalar.numel() == v.size(2) &&
             scalar.IsContiguous() &&
             (scalar.dtype() == q.dtype() ||
              scalar.dtype() == DataType::kFloat32) &&
             "CloverLocalMemory: invalid per-head scalars.");
    }
    for (const auto& tensor : {k, v, g, state, indices, valid, logit_scale,
                               history_bias, history_c, out}) {
      assert(tensor.device() == q.device() &&
             "CloverLocalMemory: device mismatch.");
    }
    assert(!out.HasBroadcastDim() && "CloverLocalMemory: overlapping output.");
  }

  virtual void operator()(const Tensor q, const Tensor k, const Tensor v,
                          const Tensor g, const Tensor state,
                          const Tensor indices, const Tensor valid,
                          const Tensor logit_scale, const Tensor history_bias,
                          const Tensor history_c, size_t block_size,
                          Tensor out) const = 0;
};

}  // namespace infini::ops
