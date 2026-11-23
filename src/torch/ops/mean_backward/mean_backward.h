#ifndef INFINI_OPS_TORCH_MEAN_BACKWARD_H_
#define INFINI_OPS_TORCH_MEAN_BACKWARD_H_

#include "base/mean_backward.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<MeanBackward, kDev, 1> : public MeanBackward {
 public:
  Operator(const Tensor grad_output, const int64_t dim, const bool keepdim,
           Tensor grad_input);

  void operator()(const Tensor grad_output, const int64_t dim,
                  const bool keepdim, Tensor grad_input) const override;
};

}  // namespace infini::ops

#endif
