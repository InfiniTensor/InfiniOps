#ifndef INFINI_OPS_BASE_MASKED_SCATTER_H_
#define INFINI_OPS_BASE_MASKED_SCATTER_H_

#include "operator.h"

namespace infini::ops {

// Replace true positions in row-major order with consecutive source values.
class MaskedScatter : public Operator<MaskedScatter> {
 public:
  MaskedScatter(const Tensor input, const Tensor mask, const Tensor source,
                Tensor out) {}
  virtual void operator()(const Tensor input, const Tensor mask,
                          const Tensor source, Tensor out) const = 0;
};

}  // namespace infini::ops

#endif
