#ifndef INFINI_OPS_BASE_PARTIAL_ROTARY_MUL_H_
#define INFINI_OPS_BASE_PARTIAL_ROTARY_MUL_H_

#include <cstdint>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class PartialRotaryMul : public Operator<PartialRotaryMul> {
 public:
  static constexpr bool kAutomaticDefault = true;

  PartialRotaryMul(Tensor x, const Tensor cos, const Tensor sin, int64_t mode,
                   int64_t slice_begin, int64_t slice_end) {}

  virtual void operator()(Tensor x, const Tensor cos, const Tensor sin,
                          int64_t mode, int64_t slice_begin,
                          int64_t slice_end) const = 0;
};

}  // namespace infini::ops

#endif
