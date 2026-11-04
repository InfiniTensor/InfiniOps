#ifndef INFINI_OPS_TORCH_REPEAT_INTERLEAVE_BACKWARD_H_
#define INFINI_OPS_TORCH_REPEAT_INTERLEAVE_BACKWARD_H_

#include "base/repeat_interleave_backward.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<RepeatInterleaveBackward, kDev, 1> : public RepeatInterleaveBackward {
 public:
  Operator(const Tensor grad_output, const int64_t dim,
                           Tensor grad_input);


  void operator()(const Tensor grad_output, const int64_t dim,
                           Tensor grad_input) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
