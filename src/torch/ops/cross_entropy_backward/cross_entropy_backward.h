#ifndef INFINI_OPS_TORCH_CROSS_ENTROPY_BACKWARD_H_
#define INFINI_OPS_TORCH_CROSS_ENTROPY_BACKWARD_H_

#include "base/cross_entropy_backward.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<CrossEntropyBackward, kDev, 1> : public CrossEntropyBackward {
 public:
  Operator(const Tensor input, const Tensor target, const Tensor grad_output,
           Tensor grad_input);

  void operator()(const Tensor input, const Tensor target,
                  const Tensor grad_output, Tensor grad_input) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_CROSS_ENTROPY_BACKWARD_H_
