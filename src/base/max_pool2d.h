#ifndef INFINI_OPS_BASE_MAX_POOL2D_H_
#define INFINI_OPS_BASE_MAX_POOL2D_H_

#include <vector>

#include "operator.h"

namespace infini::ops {
// Value-only pooling: backend-private argmax/mask layouts are not part of
// this interface. Callers needing indices use MaxPool2dWithIndices.
class MaxPool2d : public Operator<MaxPool2d> {
 public:
  MaxPool2d(const Tensor input, const std::vector<int64_t> kernel_size,
            const std::vector<int64_t> stride,
            const std::vector<int64_t> padding,
            const std::vector<int64_t> dilation, bool ceil_mode, Tensor out)
      : input_layout_{input},
        out_layout_{out},
        kernel_size_{kernel_size},
        stride_{stride},
        padding_{padding},
        dilation_{dilation},
        ceil_mode_{ceil_mode} {}
  virtual void operator()(const Tensor input,
                          const std::vector<int64_t> kernel_size,
                          const std::vector<int64_t> stride,
                          const std::vector<int64_t> padding,
                          const std::vector<int64_t> dilation, bool ceil_mode,
                          Tensor out) const = 0;

 protected:
  Tensor input_layout_, out_layout_;
  std::vector<int64_t> kernel_size_, stride_, padding_, dilation_;
  bool ceil_mode_;
};
}  // namespace infini::ops
#endif
