#ifndef INFINI_OPS_BASE_DEEPSEEK_V4_MOE_GROUPED_MATMUL_SWIGLU_QUANT_H_
#define INFINI_OPS_BASE_DEEPSEEK_V4_MOE_GROUPED_MATMUL_SWIGLU_QUANT_H_

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class DeepseekV4MoeGroupedMatmulSwigluQuant
    : public Operator<DeepseekV4MoeGroupedMatmulSwigluQuant> {
 public:
  static constexpr bool kAutomaticDefault = true;

  DeepseekV4MoeGroupedMatmulSwigluQuant(
      Tensor output, Tensor output_scale, const Tensor x,
      const Tensor weight, const Tensor weight_scale,
      const Tensor activation_scale, const Tensor group_ends,
      double swiglu_limit) {}

  virtual void operator()(Tensor output, Tensor output_scale,
                          const Tensor x, const Tensor weight,
                          const Tensor weight_scale,
                          const Tensor activation_scale,
                          const Tensor group_ends,
                          double swiglu_limit) const = 0;
};

} // namespace infini::ops

#endif
