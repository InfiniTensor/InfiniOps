#ifndef INFINI_OPS_TORCH_OPS_TRANSPOSE_BACKWARD_H_
#define INFINI_OPS_TORCH_OPS_TRANSPOSE_BACKWARD_H_

#include "base/transpose_backward.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<TransposeBackward, kDev, 1> : public TransposeBackward {
 public:
  Operator(const Tensor grad_output, const int64_t dim0, const int64_t dim1,
           Tensor grad_input);

  void operator()(const Tensor grad_output, const int64_t dim0,
                  const int64_t dim1, Tensor grad_input) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_OPS_TRANSPOSE_BACKWARD_H_