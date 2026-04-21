#ifndef INFINI_OPS_TORCH_SWIGLU_H_
#define INFINI_OPS_TORCH_SWIGLU_H_

#include "base/swiglu.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<Swiglu, kDev, 1> : public Swiglu {
 public:
  Operator(const Tensor input, const Tensor gate, Tensor out);

  void operator()(const Tensor input, const Tensor gate,
                  Tensor out) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
