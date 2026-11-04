#ifndef INFINI_OPS_TORCH_MASKED_FILL_BACKWARD_H_
#define INFINI_OPS_TORCH_MASKED_FILL_BACKWARD_H_

#include "base/masked_fill_backward.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<MaskedFillBackward, kDev, 1> : public MaskedFillBackward {
 public:
  Operator(const Tensor grad_output, const Tensor mask, Tensor grad_input);

  void operator()(const Tensor grad_output, const Tensor mask,
                  Tensor grad_input) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
