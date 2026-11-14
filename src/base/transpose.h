#ifndef INFINI_OPS_BASE_TRANSPOSE_H_
#define INFINI_OPS_BASE_TRANSPOSE_H_

#include <cassert>
#include <cstdint>
#include <initializer_list>

#include "operator.h"

namespace infini::ops {

class Transpose : public Operator<Transpose> {
 public:
  Transpose(const Tensor input, const int64_t dim0, const int64_t dim1,
            Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dim0_{NormalizeDim(dim0, input.ndim())},
        dim1_{NormalizeDim(dim1, input.ndim())},
        device_index_{out.device().index()} {
    AssertTrainingDevices({input, out});
    assert(input.ndim() >= 1 && "Transpose requires a non-scalar input.");
    assert(out.ndim() == input.ndim() && "Transpose ranks must match.");
    assert(input_type_ == out_type_ && "`Transpose` requires same dtype");
    for (Tensor::Size i = 0; i < input.ndim(); ++i) {
      const auto expect = (i == dim0_)   ? input.size(dim1_)
                          : (i == dim1_) ? input.size(dim0_)
                                         : input.size(i);
      assert(out.size(i) == expect && "`Transpose` output shape mismatch");
    }
  }

  virtual void operator()(const Tensor input, const int64_t dim0,
                          const int64_t dim1, Tensor out) const = 0;

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
  int64_t dim0_{0};
  int64_t dim1_{0};
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_TRANSPOSE_H_
