#ifndef INFINI_OPS_TORCH_MAX_POOL2D_H_
#define INFINI_OPS_TORCH_MAX_POOL2D_H_
#include "base/max_pool2d.h"

namespace infini::ops {
template <Device::Type kDev>
class Operator<MaxPool2d, kDev, 8> : public MaxPool2d {
 public:
  using MaxPool2d::MaxPool2d;
  void operator()(const Tensor input, const std::vector<int64_t> kernel_size,
                  const std::vector<int64_t> stride,
                  const std::vector<int64_t> padding,
                  const std::vector<int64_t> dilation, bool ceil_mode,
                  Tensor out) const override;
};
}  // namespace infini::ops
#endif
