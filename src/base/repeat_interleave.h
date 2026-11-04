#ifndef INFINI_OPS_BASE_REPEAT_INTERLEAVE_H_
#define INFINI_OPS_BASE_REPEAT_INTERLEAVE_H_

#include <cassert>
#include <cstdint>
#include <initializer_list>

#include "operator.h"

namespace infini::ops {

class RepeatInterleave : public Operator<RepeatInterleave> {
 public:
  RepeatInterleave(const Tensor input, const int64_t repeats, const int64_t dim,
                   Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        repeats_{repeats},
        dim_{NormalizeDim(dim, input.ndim())},
        device_index_{out.device().index()} {
    AssertTrainingDevices({input, out});
    assert(repeats_ >= 0 &&
           "`RepeatInterleave` requires non-negative repeats.");
    assert(input.ndim() == out.ndim() && "RepeatInterleave ranks must match.");
    assert(input_type_ == out_type_ &&
           "`RepeatInterleave` requires same dtype");
    for (Tensor::Size i = 0; i < input.ndim(); ++i) {
      const auto expect =
          (i == dim_) ? input.size(i) * repeats_ : input.size(i);
      assert(out.size(i) == expect &&
             "`RepeatInterleave` output shape mismatch");
    }
  }

  virtual void operator()(const Tensor input, const int64_t repeats,
                          const int64_t dim, Tensor out) const = 0;

 protected:
  static void AssertTrainingDevices(std::initializer_list<Tensor> tensors) {
    if (tensors.size() == 0) return;
    const auto device = tensors.begin()->device();
    for (const auto& tensor : tensors) {
      assert(tensor.device() == device &&
             "Training tensors must share a device type and index.");
    }
  }

  static int64_t NormalizeDim(const int64_t dim, const Tensor::Size ndim) {
    const auto rank = static_cast<int64_t>(ndim);
    assert(rank > 0 && "Training dimension requires a non-scalar tensor.");
    assert(dim >= -rank && dim < rank && "Training dimension is out of range.");
    return dim < 0 ? dim + rank : dim;
  }

  Tensor::Shape input_shape_;
  Tensor::Strides input_strides_;
  DataType input_type_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType out_type_;
  int64_t repeats_{0};
  int64_t dim_{0};
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_REPEAT_INTERLEAVE_H_
