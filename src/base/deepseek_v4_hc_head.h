#ifndef INFINI_OPS_BASE_DEEPSEEK_V4_HC_HEAD_H_
#define INFINI_OPS_BASE_DEEPSEEK_V4_HC_HEAD_H_

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class DeepseekV4HcHead : public Operator<DeepseekV4HcHead> {
 public:
  static constexpr bool kAutomaticDefault = true;

  DeepseekV4HcHead(Tensor out, const Tensor x, const Tensor hc_fn,
                   const Tensor hc_scale, const Tensor hc_base,
                   const Tensor rms_weight, double eps) {}

  virtual void operator()(Tensor out, const Tensor x, const Tensor hc_fn,
                          const Tensor hc_scale, const Tensor hc_base,
                          const Tensor rms_weight, double eps) const = 0;
};

}  // namespace infini::ops

#endif
