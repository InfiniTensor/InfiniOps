#ifndef INFINI_OPS_TORCH_SLICE_H_
#define INFINI_OPS_TORCH_SLICE_H_

#include "base/slice.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<Slice, kDev, 1> : public Slice {
 public:
  Operator(const Tensor input, const std::vector<int64_t> starts,
           const std::vector<int64_t> ends, const std::vector<int64_t> steps,
           Tensor out);

  void operator()(const Tensor input, const std::vector<int64_t> starts,
                  const std::vector<int64_t> ends,
                  const std::vector<int64_t> steps, Tensor out) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_SLICE_H_
