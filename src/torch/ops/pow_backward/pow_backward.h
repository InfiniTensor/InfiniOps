#ifndef INFINI_OPS_TORCH_POW_BACKWARD_H_
#define INFINI_OPS_TORCH_POW_BACKWARD_H_

#include "base/pow_backward.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<PowBackward, kDev, 1> : public PowBackward {
 public:
  Operator(const Tensor grad_output, const Tensor input, const double scalar,
              const bool scalar_is_base, Tensor grad_input);

  void operator()(const Tensor grad_output, const Tensor input, const double scalar,
              const bool scalar_is_base, Tensor grad_input) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
