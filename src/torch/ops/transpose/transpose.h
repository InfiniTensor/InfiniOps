#ifndef INFINI_OPS_TORCH_TRANSPOSE_H_
#define INFINI_OPS_TORCH_TRANSPOSE_H_

#include "base/transpose.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<Transpose, kDev, 1> : public Transpose {
 public:
  Operator(const Tensor input, const int64_t dim0, const int64_t dim1,
           Tensor out);

  void operator()(const Tensor input, const int64_t dim0, const int64_t dim1,
                  Tensor out) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
