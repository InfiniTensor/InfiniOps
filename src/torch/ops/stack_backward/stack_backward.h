#ifndef INFINI_OPS_TORCH_STACK_BACKWARD_H_
#define INFINI_OPS_TORCH_STACK_BACKWARD_H_

#include "base/stack_backward.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<StackBackward, kDev, 1> : public StackBackward {
 public:
    Operator(const Tensor grad_output, const int64_t dim,
                std::vector<Tensor> outs);

    void operator()(const Tensor grad_output, const int64_t dim,
                std::vector<Tensor> outs) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
