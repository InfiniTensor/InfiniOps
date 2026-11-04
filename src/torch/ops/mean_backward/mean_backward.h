#ifndef INFINI_OPS_TORCH_mean_backward_H_
#define INFINI_OPS_TORCH_mean_backward_H_

#include "base/mean_backward.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<MeanBackward, kDev, 1> : public MeanBackward {
 public:
  Operator(const Tensor grad_output, const int64_t dim, const bool keep_dim,
               Tensor grad_input);


  void operator()(const Tensor grad_output, const int64_t dim, const bool keep_dim,
               Tensor grad_input) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
