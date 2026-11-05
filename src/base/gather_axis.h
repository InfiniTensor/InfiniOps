#ifndef INFINI_OPS_BASE_GATHER_AXIS_H_
#define INFINI_OPS_BASE_GATHER_AXIS_H_

#include "operator.h"

namespace infini::ops {

// Select a one-dimensional list of indices, wrapping negatives on the axis.
class GatherAxis : public Operator<GatherAxis> {
 public:
  GatherAxis(const Tensor input, const Tensor index, const int64_t dim,
             Tensor out) {}

  virtual void operator()(const Tensor input, const Tensor index,
                          const int64_t dim, Tensor out) const = 0;
};

}  // namespace infini::ops

#endif
