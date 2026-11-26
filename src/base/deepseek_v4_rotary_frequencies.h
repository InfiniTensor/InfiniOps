#pragma once
#include "operator.h"
namespace infini::ops {
class DeepseekV4RotaryFrequencies : public Operator<DeepseekV4RotaryFrequencies> {
 public:
  DeepseekV4RotaryFrequencies(Tensor out,double theta,bool scaled,double low,double high,
                            double factor,bool split_cos_sin) {
    (void)out;(void)theta;(void)scaled;(void)low;(void)high;(void)factor;(void)split_cos_sin;
  }
  virtual void operator()(Tensor out,double theta,bool scaled,double low,double high,
                          double factor,bool split_cos_sin) const = 0;
};
}  // namespace infini::ops
