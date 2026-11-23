#ifndef INFINI_OPS_BASE_HC_POST_H_
#define INFINI_OPS_BASE_HC_POST_H_

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class HcPost : public Operator<HcPost> {
 public:
  static constexpr bool kAutomaticDefault = true;

  HcPost(Tensor out, const Tensor x, const Tensor residual, const Tensor post,
         const Tensor comb) {}

  virtual void operator()(Tensor out, const Tensor x, const Tensor residual,
                          const Tensor post, const Tensor comb) const = 0;
};

}  // namespace infini::ops

#endif
