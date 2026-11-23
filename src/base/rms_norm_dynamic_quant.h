#ifndef INFINI_OPS_BASE_RMS_NORM_DYNAMIC_QUANT_H_
#define INFINI_OPS_BASE_RMS_NORM_DYNAMIC_QUANT_H_

#include <cstdint>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class RmsNormDynamicQuant : public Operator<RmsNormDynamicQuant> {
 public:
  static constexpr bool kAutomaticDefault = true;

  RmsNormDynamicQuant(Tensor packed, Tensor packed2, Tensor scale,
                      Tensor scale2, const Tensor input, const Tensor gamma,
                      double epsilon) {}

  virtual void operator()(Tensor packed, Tensor packed2, Tensor scale,
                          Tensor scale2, const Tensor input,
                          const Tensor gamma, double epsilon) const = 0;
};

}  // namespace infini::ops

#endif
