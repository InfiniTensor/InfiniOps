#ifndef INFINI_OPS_TORCH_ADD_RMS_NORM_H_
#define INFINI_OPS_TORCH_ADD_RMS_NORM_H_

#include "base/add_rms_norm.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<AddRmsNorm, kDev, 1> : public AddRmsNorm {
 public:
  Operator(const Tensor input, const Tensor other, const Tensor weight,
           float eps, Tensor out, Tensor residual_out);

  void operator()(const Tensor input, const Tensor other, const Tensor weight,
                  float eps, Tensor out, Tensor residual_out) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
