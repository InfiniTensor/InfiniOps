#pragma once
#include "operator.h"

namespace infini::ops {

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
