#ifndef INFINI_OPS_TORCH_MUL_BACKWARD_H_
#define INFINI_OPS_TORCH_MUL_BACKWARD_H_

#include "base/mul_backward.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<MulBackward, kDev, 1> : public MulBackward {
 public:
  Operator(const Tensor grad_output, const Tensor a, const Tensor b,
           Tensor grad_a, Tensor grad_b);

  void operator()(const Tensor grad_output, const Tensor a, const Tensor b,
                  Tensor grad_a, Tensor grad_b) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_MUL_BACKWARD_H_