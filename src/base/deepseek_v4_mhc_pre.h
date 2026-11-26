#pragma once
#include "operator.h"

namespace infini::ops {
class DeepseekV4MhcPre : public Operator<DeepseekV4MhcPre> {
 public:
  DeepseekV4MhcPre(const Tensor residual,const Tensor projection,const Tensor scales,
                  const Tensor bias,double norm_eps,double mix_eps,int64_t iterations,
                  Tensor collapsed,Tensor post,Tensor combination) {}
  virtual void operator()(const Tensor residual,const Tensor projection,const Tensor scales,
                          const Tensor bias,double norm_eps,double mix_eps,int64_t iterations,
                          Tensor collapsed,Tensor post,Tensor combination) const = 0;
};
}
