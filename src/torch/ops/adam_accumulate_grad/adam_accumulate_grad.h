#ifndef INFINI_OPS_TORCH_ADAM_ACCUMULATE_GRAD_H_
#define INFINI_OPS_TORCH_ADAM_ACCUMULATE_GRAD_H_

#include "base/adam_accumulate_grad.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<AdamAccumulateGrad, kDev, 1> : public AdamAccumulateGrad {
 public:
  using AdamAccumulateGrad::operator();

  Operator(const Tensor grad, const double learning_rate, const double beta1,
           const double beta2, const double eps, const int64_t step,
           Tensor param, Tensor m, Tensor v);

  [[deprecated("Use grad, attributes, param, m, v argument order.")]]
  Operator(const Tensor grad, Tensor param, Tensor m, Tensor v,
           const double learning_rate, const double beta1, const double beta2,
           const double eps, const int64_t step);

  void operator()(const Tensor grad, const double learning_rate,
                  const double beta1, const double beta2, const double eps,
                  const int64_t step, Tensor param, Tensor m,
                  Tensor v) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_ADAM_ACCUMULATE_GRAD_H_
