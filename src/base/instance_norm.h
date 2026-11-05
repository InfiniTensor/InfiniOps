#ifndef INFINI_OPS_BASE_INSTANCE_NORM_H_
#define INFINI_OPS_BASE_INSTANCE_NORM_H_

#include "operator.h"

namespace infini::ops {

class InstanceNorm : public Operator<InstanceNorm> {
 public:
  InstanceNorm(const Tensor input, const Tensor scale, const Tensor bias,
               float epsilon, Tensor out) {}
  virtual void operator()(const Tensor input, const Tensor scale, const Tensor bias,
                          float epsilon, Tensor out) const = 0;
};

}  // namespace infini::ops

#endif
