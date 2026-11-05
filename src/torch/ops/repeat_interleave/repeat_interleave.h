#ifndef INFINI_OPS_TORCH_REPEAT_INTERLEAVE_H_
#define INFINI_OPS_TORCH_REPEAT_INTERLEAVE_H_

#include "base/repeat_interleave.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<RepeatInterleave, kDev, 1> : public RepeatInterleave {
 public:
  Operator(const Tensor input, const int64_t repeats, const int64_t dim,
           Tensor out);

  void operator()(const Tensor input, const int64_t repeats, const int64_t dim,
                  Tensor out) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
