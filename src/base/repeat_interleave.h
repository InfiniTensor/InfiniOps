#ifndef INFINI_OPS_BASE_REPEAT_INTERLEAVE_H_
#define INFINI_OPS_BASE_REPEAT_INTERLEAVE_H_

#include <cassert>
#include <cstdint>

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
    assert(repeats_ > 0 && "`RepeatInterleave` requires positive `repeats`");
    assert(input_type_ == out_type_ && "`RepeatInterleave` requires same dtype");
    for (Tensor::Size i = 0; i < input.ndim(); ++i) {
      const auto expect = (i == dim_) ? input.size(i) * repeats_ : input.size(i);
      assert(out.size(i) == expect && "`RepeatInterleave` output shape mismatch");
    }
  }

  virtual void operator()(const Tensor input, const int64_t repeats,
                          const int64_t dim, Tensor out) const = 0;

 protected:
  static int64_t NormalizeDim(const int64_t dim, const Tensor::Size ndim) {
    assert(ndim > 0 && "`RepeatInterleave` requires ndim > 0");
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
  int64_t repeats_{0};
  int64_t dim_{0};
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_REPEAT_INTERLEAVE_H_
