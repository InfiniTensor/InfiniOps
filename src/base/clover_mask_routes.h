#pragma once
#include "operator.h"

namespace infini::ops {

// Clover-specific interface of the expert-budget decoder, used through the
// C++ API. Every expert ID of `ids` (`[rows, topk]`) on a row that is not live
// is replaced by the out-of-range sentinel `num_experts`, which must be the
// expert count of the routed MoE so that its alignment drops the route. `out`
// may be `ids`.
class CloverMaskRoutes : public Operator<CloverMaskRoutes> {
 public:
  CloverMaskRoutes(const Tensor ids, const Tensor live, int64_t num_experts,
                   Tensor out) {
    assert(ids.ndim() == 2 && ids.IsContiguous() && live.IsContiguous() &&
           out.IsContiguous() && live.numel() == ids.size(0) &&
           out.shape() == ids.shape() && ids.dtype() == DataType::kInt32 &&
           out.dtype() == ids.dtype() && live.dtype() == DataType::kInt8 &&
           ids.device() == live.device() && ids.device() == out.device() &&
           "CloverMaskRoutes: invalid routes.");
  }

  virtual void operator()(const Tensor ids, const Tensor live,
                          int64_t num_experts, Tensor out) const = 0;
};

}  // namespace infini::ops
