#pragma once
#include "operator.h"

namespace infini::ops {
class DeepseekV4MhcPost : public Operator<DeepseekV4MhcPost> {
 public:
  DeepseekV4MhcPost(const Tensor activation,const Tensor residual,const Tensor post,
                   const Tensor combination,Tensor out) {}
  virtual void operator()(const Tensor activation,const Tensor residual,const Tensor post,
                          const Tensor combination,Tensor out) const = 0;
};
}
