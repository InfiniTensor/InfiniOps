#ifndef INFINI_OPS_BASE_SLICE_BACKWARD_H_
#define INFINI_OPS_BASE_SLICE_BACKWARD_H_

#include <cassert>
#include <cstdint>
#include <vector>

#include "operator.h"

namespace infini::ops {

class SliceBackward : public Operator<SliceBackward> {
 public:
  SliceBackward(const Tensor grad_output, const std::vector<int64_t> starts,
                const std::vector<int64_t> ends,
                const std::vector<int64_t> steps, Tensor grad_input)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        starts_{starts},
        ends_{ends},
        steps_{steps},
        device_index_{grad_input.device().index()} {
    assert(starts_.size() == grad_input.ndim() &&
           "`SliceBackward` requires starts/ends/steps to match `grad_input` rank");
    assert(grad_output.ndim() == grad_input.ndim() &&
           "`SliceBackward` requires matching rank");
    assert(grad_output_type_ == grad_input_type_ &&
           "`SliceBackward` requires same dtype");
    for (Tensor::Size i = 0; i < grad_input.ndim(); ++i) {
      assert(steps_[i] > 0 && "`SliceBackward` requires positive steps");
      assert(starts_[i] <= ends_[i] && "`SliceBackward` requires start <= end");
      const auto expected = (ends_[i] - starts_[i] + steps_[i] - 1) / steps_[i];
      assert(grad_output.size(i) == expected &&
             "`SliceBackward` grad_output dim must equal the slice dim");
    }
  }

  virtual void operator()(const Tensor grad_output,
                          const std::vector<int64_t> starts,
                          const std::vector<int64_t> ends,
                          const std::vector<int64_t> steps,
                          Tensor grad_input) const = 0;

 protected:
  Tensor::Shape grad_output_shape_;
  Tensor::Strides grad_output_strides_;
  DataType grad_output_type_;
  Tensor::Shape grad_input_shape_;
  Tensor::Strides grad_input_strides_;
  DataType grad_input_type_;
  std::vector<int64_t> starts_;
  std::vector<int64_t> ends_;
  std::vector<int64_t> steps_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_SLICE_BACKWARD_H_