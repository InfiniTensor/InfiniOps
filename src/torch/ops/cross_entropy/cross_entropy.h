#ifndef INFINI_OPS_TORCH_CROSS_ENTROPY_H_
#define INFINI_OPS_TORCH_CROSS_ENTROPY_H_

#include "base/cross_entropy.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<CrossEntropy, kDev, 1> : public CrossEntropy {
 public:
  Operator(const Tensor input, const Tensor target,
           std::optional<Tensor> weight, int64_t ignore_index,
           std::string reduction, double label_smoothing, Tensor out);

  void operator()(const Tensor input, const Tensor target,
                  std::optional<Tensor> weight, int64_t ignore_index,
                  std::string reduction, double label_smoothing,
                  Tensor out) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_CROSS_ENTROPY_H_
