#ifndef INFINI_OPS_TORCH_RANDOM_SAMPLE_H_
#define INFINI_OPS_TORCH_RANDOM_SAMPLE_H_

#include "base/random_sample.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<RandomSample, kDev, 1> : public RandomSample {
 public:
  Operator(const Tensor logits, float random_val, float topp, int topk,
           float temperature, Tensor out);

  void operator()(const Tensor logits, float random_val, float topp, int topk,
                  float temperature, Tensor out) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
