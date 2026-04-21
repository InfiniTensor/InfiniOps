#ifndef INFINI_OPS_TORCH_RMS_NORM_H_
#define INFINI_OPS_TORCH_RMS_NORM_H_

#include "base/rms_norm.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<RmsNorm, kDev, 1> : public RmsNorm {
 public:
  Operator(const Tensor input, const Tensor weight, float eps, Tensor out);

  using RmsNorm::operator();

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
