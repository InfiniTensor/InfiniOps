#ifndef INFINI_OPS_TRITON_OPS_ADD_JIT_H_
#define INFINI_OPS_TRITON_OPS_ADD_JIT_H_

#include "base/add.h"
#include "triton/jit/jit.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<Add, kDev, 10> : public triton::jit::OperatorBase<Add, kDev> {
 public:
  using triton::jit::OperatorBase<Add, kDev>::OperatorBase;

  void operator()(const Tensor input, const Tensor other, const double alpha,
                  Tensor out) const;
};

}  // namespace infini::ops

#endif
