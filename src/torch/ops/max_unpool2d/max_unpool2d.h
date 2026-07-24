#ifndef INFINI_OPS_TORCH_MAX_UNPOOL2D_H_
#define INFINI_OPS_TORCH_MAX_UNPOOL2D_H_

#include "base/max_unpool2d.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<MaxUnpool2d, kDev, 8> : public MaxUnpool2d {
 public:
  using MaxUnpool2d::MaxUnpool2d;

 protected:
  void Run(const Tensor input, const Tensor indices,
           const std::vector<int64_t> output_size, Tensor out) const override;
};

}  // namespace infini::ops

#endif
