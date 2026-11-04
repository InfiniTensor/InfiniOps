#ifndef INFINI_OPS_TORCH_ADD_BACKWARD_H_
#define INFINI_OPS_TORCH_ADD_BACKWARD_H_

#include "base/add_backward.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<AddBackward, kDev, 1> : public AddBackward {
 public:
  Operator(const Tensor grad_output, Tensor grad_a, Tensor grad_b);

  void operator()(const Tensor grad_output, Tensor grad_a,
                  Tensor grad_b) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_ADD_BACKWARD_H_