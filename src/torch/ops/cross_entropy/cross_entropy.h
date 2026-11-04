#ifndef INFINI_OPS_TORCH_CROSS_ENTROPY_H_
#define INFINI_OPS_TORCH_CROSS_ENTROPY_H_

#include "base/cross_entropy.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<CrossEntropy, kDev, 1> : public CrossEntropy {
 public:
  Operator(const Tensor input, const Tensor target, Tensor out);


  void operator()(const Tensor input, const Tensor target, Tensor out) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
