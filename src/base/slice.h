#ifndef INFINI_OPS_BASE_SLICE_H_
#define INFINI_OPS_BASE_SLICE_H_

#include <cassert>
#include <cstdint>
#include <vector>

#include "operator.h"

namespace infini::ops {

class Slice : public Operator<Slice> {
 public:
  Slice(const Tensor input, const std::vector<int64_t> starts,
        const std::vector<int64_t> ends, const std::vector<int64_t> steps,
        Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        starts_{starts},
        ends_{ends},
        steps_{steps},
        device_index_{out.device().index()} {
    assert(starts_.size() == input.ndim() && ends_.size() == input.ndim() &&
           steps_.size() == input.ndim() &&
           "`Slice` requires starts/ends/steps to match input rank");
    assert(out.ndim() == input.ndim() && "`Slice` requires matching rank");
    assert(input_type_ == out_type_ && "`Slice` requires same dtype");
    for (Tensor::Size i = 0; i < input.ndim(); ++i) {
      assert(steps_[i] > 0 && "`Slice` requires positive steps (InfiniTrain semantics)");
      assert(starts_[i] <= ends_[i] && "`Slice` requires start <= end");
      const auto expected = (ends_[i] - starts_[i] + steps_[i] - 1) / steps_[i];
      assert(out.size(i) == expected &&
             "`Slice` output dim must equal ceil((end - start) / step)");
    }
  }

  virtual void operator()(const Tensor input, const std::vector<int64_t> starts,
                          const std::vector<int64_t> ends,
                          const std::vector<int64_t> steps, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;
  Tensor::Strides input_strides_;
  DataType input_type_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType out_type_;
  std::vector<int64_t> starts_;
  std::vector<int64_t> ends_;
  std::vector<int64_t> steps_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_SLICE_H_