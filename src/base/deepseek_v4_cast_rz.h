#pragma once

#include "operator.h"

namespace infini::ops {
// Explicit HCU writeback boundary for FP32 TP sums. Other casts retain their
// normal rounding semantics.
class DeepseekV4CastRz : public Operator<DeepseekV4CastRz> {
 public:
  DeepseekV4CastRz(const Tensor input, Tensor out) { (void)input; (void)out; }
  virtual void operator()(const Tensor input, Tensor out) const = 0;
};
}  // namespace infini::ops
