#ifndef INFINI_OPS_TORCH_SUB_BACKWARD_H_
#define INFINI_OPS_TORCH_SUB_BACKWARD_H_

#include "base/sub_backward.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<SubBackward, kDev, 1> : public SubBackward {
 public:
  Operator(const Tensor grad_output, Tensor grad_a, Tensor grad_b);

  void operator()(const Tensor grad_output, Tensor grad_a,
                  Tensor grad_b) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_SUB_BACKWARD_H_