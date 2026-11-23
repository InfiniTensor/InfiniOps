#ifndef INFINI_OPS_BASE_SCATTER_ND_UPDATE_V2_H_
#define INFINI_OPS_BASE_SCATTER_ND_UPDATE_V2_H_

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class ScatterNdUpdateV2 : public Operator<ScatterNdUpdateV2> {
 public:
  static constexpr bool kAutomaticDefault = true;

  ScatterNdUpdateV2(Tensor var, const Tensor indices, const Tensor updates) {}

  virtual void operator()(Tensor var, const Tensor indices,
                          const Tensor updates) const = 0;
};

}  // namespace infini::ops

#endif
