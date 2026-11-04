#ifndef INFINI_OPS_TORCH_SPLIT_H_
#define INFINI_OPS_TORCH_SPLIT_H_

#include "base/split.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<Split, kDev, 1> : public Split {
 public:
  Operator(const Tensor input, const int64_t split_size, const int64_t dim,
           std::vector<Tensor> outs);

  void operator()(const Tensor input, const int64_t split_size,
                  const int64_t dim, std::vector<Tensor> outs) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
