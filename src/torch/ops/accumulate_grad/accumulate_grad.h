#ifndef INFINI_OPS_TORCH_ACCUMULATE_GRAD_H_
#define INFINI_OPS_TORCH_ACCUMULATE_GRAD_H_

#include "base/accumulate_grad.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<AccumulateGrad, kDev, 1> : public AccumulateGrad {
 public:
  Operator(const Tensor gradient, const double rate, Tensor tensor);

  void operator()(const Tensor gradient, const double rate,
                  Tensor tensor) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
