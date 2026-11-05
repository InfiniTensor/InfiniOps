#ifndef INFINI_OPS_BASE_SLICE_H_
#define INFINI_OPS_BASE_SLICE_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
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
    AssertTrainingDevices({input, out});
    assert(starts_.size() == input.ndim() && ends_.size() == input.ndim() &&
           steps_.size() == input.ndim() &&
           "`Slice` requires starts/ends/steps to match input rank");
    assert(out.ndim() == input.ndim() && "`Slice` requires matching rank");
    assert(input_type_ == out_type_ && "`Slice` requires same dtype");
    for (Tensor::Size i = 0; i < input.ndim(); ++i) {
      assert(steps_[i] > 0 && "`Slice` requires positive steps");
      const auto expected =
          TrainingSliceLength(starts_[i], ends_[i], steps_[i], input.size(i));
      assert(out.size(i) == expected &&
             "`Slice` output dim must equal ceil((end - start) / step)");
    }
  }

  virtual void operator()(const Tensor input, const std::vector<int64_t> starts,
                          const std::vector<int64_t> ends,
                          const std::vector<int64_t> steps,
                          Tensor out) const = 0;

 protected:
  static void AssertTrainingDevices(std::initializer_list<Tensor> tensors) {
    if (tensors.size() == 0) return;
    const auto device = tensors.begin()->device();
    for (const auto& tensor : tensors) {
      assert(tensor.device() == device &&
             "Training tensors must share a device type and index.");
    }
  }

  static int64_t TrainingSliceLength(int64_t start, int64_t end, int64_t step,
                                     Tensor::Size size) {
    assert(step > 0 && "Training slices require a positive step.");
    const auto length = static_cast<int64_t>(size);
    const auto normalize = [length](int64_t index) {
      if (index < 0) index += length;
      return std::clamp<int64_t>(index, 0, length);
    };
    start = normalize(start);
    end = normalize(end);
    return start >= end ? 0 : 1 + (end - start - 1) / step;
  }

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
