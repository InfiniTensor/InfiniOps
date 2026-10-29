#pragma once

#include "operator.h"

namespace infini::ops {

class CloverRecurrentGdn : public Operator<CloverRecurrentGdn> {
 public:
  CloverRecurrentGdn(const Tensor q, const Tensor k, const Tensor v,
                     const Tensor g, const Tensor beta, const Tensor offsets,
                     const Tensor initial_indices, const Tensor final_indices,
                     Tensor state, Tensor out) {
    assert((q.ndim() == 4 && q.size(0) == 1 && k.shape() == q.shape() &&
            v.ndim() == 4 && v.size(0) == 1 && q.size(1) == v.size(1) &&
            q.size(2) > 0 && v.size(2) % q.size(2) == 0 && q.size(3) == 128 &&
            v.size(3) == 128 && out.shape() == v.shape() &&
            g.shape() == Tensor::Shape{1, q.size(1), v.size(2)} &&
            beta.shape() == g.shape() && state.ndim() == 4 &&
            state.size(1) == v.size(2) && state.size(2) == 128 &&
            state.size(3) == 128 && offsets.ndim() == 1 &&
            offsets.numel() >= 2 && initial_indices.ndim() == 1 &&
            final_indices.shape() == initial_indices.shape() &&
            initial_indices.numel() + 1 == offsets.numel()) &&
           "CloverRecurrentGdn: invalid sequence or state shape.");
    assert(
        (q.dtype() == DataType::kBFloat16 || q.dtype() == DataType::kFloat16) &&
        k.dtype() == q.dtype() && v.dtype() == q.dtype() &&
        out.dtype() == q.dtype() && state.dtype() == DataType::kFloat32 &&
        g.dtype() == DataType::kFloat32 && beta.dtype() == DataType::kFloat32 &&
        "CloverRecurrentGdn: invalid dtype.");
    for (const auto& index : {offsets, initial_indices, final_indices}) {
      assert(index.IsContiguous() &&
             (index.dtype() == DataType::kInt32 ||
              index.dtype() == DataType::kInt64) &&
             "CloverRecurrentGdn: invalid sequence indices.");
    }
    for (const auto& tensor :
         {k, v, g, beta, offsets, initial_indices, final_indices, state, out}) {
      assert(tensor.device() == q.device() &&
             "CloverRecurrentGdn: device mismatch.");
    }
  }

  virtual void operator()(const Tensor q, const Tensor k, const Tensor v,
                          const Tensor g, const Tensor beta,
                          const Tensor offsets, const Tensor initial_indices,
                          const Tensor final_indices, Tensor state,
                          Tensor out) const = 0;
};

}  // namespace infini::ops
