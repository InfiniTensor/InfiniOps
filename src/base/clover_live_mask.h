#pragma once
#include "operator.h"

namespace infini::ops {

// Clover-specific interface of the expert-budget decoder, used through the
// C++ API. `ids` holds the token IDs of the current preview and `previous`
// the snapshot of the preceding one; a row is live when either ID is
// `mask_id` or when `refresh` is set. `live` receives 1 or 0 per row and
// `previous` is overwritten with `ids` for the next call. `previous` must be a
// buffer of its own; nothing else is written.
class CloverLiveMask : public Operator<CloverLiveMask> {
 public:
  CloverLiveMask(const Tensor ids, Tensor previous, int64_t mask_id,
                 bool refresh, Tensor live) {
    assert(ids.IsContiguous() && previous.IsContiguous() &&
           live.IsContiguous() && ids.numel() == previous.numel() &&
           ids.numel() == live.numel() && ids.dtype() == DataType::kInt64 &&
           previous.dtype() == ids.dtype() && live.dtype() == DataType::kInt8 &&
           ids.device() == previous.device() && ids.device() == live.device() &&
           "CloverLiveMask: invalid token state.");
  }

  virtual void operator()(const Tensor ids, Tensor previous, int64_t mask_id,
                          bool refresh, Tensor live) const = 0;
};

}  // namespace infini::ops
