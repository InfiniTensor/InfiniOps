#ifndef INFINI_OPS_TORCH_ADAM_ACCUMULATE_GRAD_H_
#define INFINI_OPS_TORCH_ADAM_ACCUMULATE_GRAD_H_

#include "base/adam_accumulate_grad.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<AdamAccumulateGrad, kDev, 1> : public AdamAccumulateGrad {
 public:
  Operator(const Tensor grad, const double learning_rate, const double beta1,
           const double beta2, const double eps, const int64_t step,
           Tensor param, Tensor m, Tensor v);

  void operator()(const Tensor grad, const double learning_rate,
                  const double beta1, const double beta2, const double eps,
                  const int64_t step, Tensor param, Tensor m,
                  Tensor v) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_ADAM_ACCUMULATE_GRAD_H_
