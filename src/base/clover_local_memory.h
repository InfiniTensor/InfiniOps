#pragma once

#include "operator.h"

namespace infini::ops {

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
