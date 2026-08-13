#ifndef INFINI_OPS_TRITON_OPS_ADD_JIT_H_
#define INFINI_OPS_TRITON_OPS_ADD_JIT_H_

#include "base/add.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kNvidia, 10> : public Add {
 public:
  using Add::Add;

  void operator()(const Tensor input, const Tensor other, const double alpha,
                  Tensor out) const override;
};

}  // namespace infini::ops

#endif
