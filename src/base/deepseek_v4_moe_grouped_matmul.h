#ifndef INFINI_OPS_BASE_DEEPSEEK_V4_MOE_GROUPED_MATMUL_H_
#define INFINI_OPS_BASE_DEEPSEEK_V4_MOE_GROUPED_MATMUL_H_

#include <optional>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class DeepseekV4MoeGroupedMatmul
    : public Operator<DeepseekV4MoeGroupedMatmul> {
 public:
  static constexpr bool kAutomaticDefault = true;

  DeepseekV4MoeGroupedMatmul(Tensor out, const Tensor x, const Tensor weight,
                             const std::optional<Tensor> weight_scale,
                             const std::optional<Tensor> per_token_scale,
                             const Tensor group_list,
                             int64_t group_list_type) {}

  virtual void operator()(Tensor out, const Tensor x, const Tensor weight,
                          const std::optional<Tensor> weight_scale,
                          const std::optional<Tensor> per_token_scale,
                          const Tensor group_list,
                          int64_t group_list_type) const = 0;
};

}  // namespace infini::ops

#endif
