#pragma once
#include "operator.h"

namespace infini::ops {
class DeepseekV4Route : public Operator<DeepseekV4Route> {
 public:
  DeepseekV4Route(const Tensor logits,const std::optional<Tensor> bias,
                  const std::optional<Tensor> tokens,const std::optional<Tensor> hash,
                  double scale,Tensor weights,Tensor indices) {}
  virtual void operator()(const Tensor logits,const std::optional<Tensor> bias,
                          const std::optional<Tensor> tokens,const std::optional<Tensor> hash,
                          double scale,Tensor weights,Tensor indices) const = 0;
};
}
