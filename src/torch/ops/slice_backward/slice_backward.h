#ifndef INFINI_OPS_TORCH_SLICE_BACKWARD_H_
#define INFINI_OPS_TORCH_SLICE_BACKWARD_H_

#include "base/slice_backward.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<SliceBackward, kDev, 1> : public SliceBackward {
 public:
  Operator(const Tensor grad_output, const std::vector<int64_t> starts,
           const std::vector<int64_t> ends, const std::vector<int64_t> steps,
           Tensor grad_input);

  void operator()(const Tensor grad_output, const std::vector<int64_t> starts,
                  const std::vector<int64_t> ends,
                  const std::vector<int64_t> steps,
                  Tensor grad_input) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
