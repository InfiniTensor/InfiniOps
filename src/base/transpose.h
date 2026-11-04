#ifndef INFINI_OPS_BASE_TRANSPOSE_H_
#define INFINI_OPS_BASE_TRANSPOSE_H_

#include <cassert>
#include <cstdint>

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
    assert(input.ndim() >= 2 && "`Transpose` requires at least 2 dims");
    assert(dim0_ != dim1_ && "`Transpose` requires distinct dims");
    assert(input_type_ == out_type_ && "`Transpose` requires same dtype");
    for (Tensor::Size i = 0; i < input.ndim(); ++i) {
      const auto expect = (i == dim0_) ? input.size(dim1_)
                          : (i == dim1_) ? input.size(dim0_)
                                         : input.size(i);
      assert(out.size(i) == expect && "`Transpose` output shape mismatch");
    }
  }

  virtual void operator()(const Tensor input, const int64_t dim0,
                          const int64_t dim1, Tensor out) const = 0;

 protected:
  static int64_t NormalizeDim(const int64_t dim, const Tensor::Size ndim) {
    assert(ndim > 0 && "`Transpose` requires ndim > 0");
    if (ndim == 0) return 0;
    const auto d = dim % static_cast<int64_t>(ndim);
    return d < 0 ? d + static_cast<int64_t>(ndim) : d;
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
