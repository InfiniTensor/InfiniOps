#pragma once
#include "operator.h"

namespace infini::ops {

// Clover-specific interface of the expert-budget decoder, used through the
// C++ API. A live row takes its fresh routed output, any other row the output
// cached by the preceding call. `out` receives the selection and `cache` is
// updated with it. `out` may alias `fresh` or `cache`; `fresh` and `cache`
// must not alias each other.
class CloverMergeMoe : public Operator<CloverMergeMoe> {
 public:
  CloverMergeMoe(const Tensor fresh, const Tensor live, Tensor cache,
                 Tensor out) {
    assert(fresh.ndim() == 2 && live.numel() == fresh.size(0) &&
           cache.shape() == fresh.shape() && out.shape() == fresh.shape() &&
           cache.dtype() == fresh.dtype() && out.dtype() == fresh.dtype() &&
           live.dtype() == DataType::kInt8 && fresh.IsContiguous() &&
           cache.IsContiguous() && out.IsContiguous() && live.IsContiguous() &&
           fresh.device() == live.device() &&
           fresh.device() == cache.device() && fresh.device() == out.device() &&
           "CloverMergeMoe: invalid cached rows.");
  }

  virtual void operator()(const Tensor fresh, const Tensor live, Tensor cache,
                          Tensor out) const = 0;
};

}  // namespace infini::ops
