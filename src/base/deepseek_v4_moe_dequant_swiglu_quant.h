#ifndef INFINI_OPS_BASE_DEEPSEEK_V4_MOE_DEQUANT_SWIGLU_QUANT_H_
#define INFINI_OPS_BASE_DEEPSEEK_V4_MOE_DEQUANT_SWIGLU_QUANT_H_

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class DeepseekV4MoeDequantSwigluQuant
    : public Operator<DeepseekV4MoeDequantSwigluQuant> {
 public:
  static constexpr bool kAutomaticDefault = true;

  DeepseekV4MoeDequantSwigluQuant(Tensor out, Tensor out_scale,
                                  const Tensor x, const Tensor weight_scale,
                                  const Tensor activation_scale,
                                  const Tensor group_index) {}

  virtual void operator()(Tensor out, Tensor out_scale, const Tensor x,
                          const Tensor weight_scale,
                          const Tensor activation_scale,
                          const Tensor group_index) const = 0;
};

}  // namespace infini::ops

#endif
