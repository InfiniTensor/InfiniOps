#ifndef INFINI_OPS_BASE_DEEPSEEK_V4_MOE_TOKEN_UNPERMUTE_H_
#define INFINI_OPS_BASE_DEEPSEEK_V4_MOE_TOKEN_UNPERMUTE_H_

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class DeepseekV4MoeTokenUnpermute
    : public Operator<DeepseekV4MoeTokenUnpermute> {
 public:
  static constexpr bool kAutomaticDefault = true;

  DeepseekV4MoeTokenUnpermute(Tensor out, const Tensor permuted_tokens,
                              const Tensor sorted_indices,
                              const Tensor probs) {}

  virtual void operator()(Tensor out, const Tensor permuted_tokens,
                          const Tensor sorted_indices,
                          const Tensor probs) const = 0;
};

}  // namespace infini::ops

#endif
