#ifndef INFINI_OPS_TORCH_RSQRT_BACKWARD_H_
#define INFINI_OPS_TORCH_RSQRT_BACKWARD_H_

#include "base/rsqrt_backward.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<RsqrtBackward, kDev, 1> : public RsqrtBackward {
 public:
  Operator(const Tensor grad_output, const Tensor input, Tensor grad_input);


  void operator()(const Tensor grad_output, const Tensor input, Tensor grad_input) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
