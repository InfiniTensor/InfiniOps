#ifndef INFINI_OPS_TORCH_MUL_BACKWARD_H_
#define INFINI_OPS_TORCH_MUL_BACKWARD_H_

#include "base/mul_backward.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<MulBackward, kDev, 1> : public MulBackward {
 public:
  Operator(const Tensor grad_output, const Tensor input, const Tensor other,
           Tensor grad_input, Tensor grad_other);

  void operator()(const Tensor grad_output, const Tensor input,
                  const Tensor other, Tensor grad_input,
                  Tensor grad_other) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_MUL_BACKWARD_H_
