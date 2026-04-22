#ifndef INFINI_OPS_BASE_SILU_AND_MUL_H_
#define INFINI_OPS_BASE_SILU_AND_MUL_H_

#include "operator.h"

namespace infini::ops {

class SiluAndMul : public Operator<SiluAndMul> {
 public:
  SiluAndMul(const Tensor x, int64_t dim, Tensor out)
      : x_shape_{x.shape()},
        x_strides_{x.strides()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        x_dtype_{x.dtype()},
        out_dtype_{out.dtype()},
        dim_{dim},
        ndim_{x.ndim()},
        is_x_contiguous_{x.IsContiguous()},
        is_out_contiguous_{out.IsContiguous()} {
    assert(x_dtype_ == out_dtype_ &&
           "`SiluAndMul`: `x` and `out` must have the same dtype.");
  }

  virtual void operator()(const Tensor x, int64_t dim, Tensor out) const = 0;

 protected:
  Tensor::Shape x_shape_;

  Tensor::Strides x_strides_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  const DataType x_dtype_;

  const DataType out_dtype_;

  int64_t dim_;

  Tensor::Size ndim_;

  bool is_x_contiguous_;

  bool is_out_contiguous_;
};

}  // namespace infini::ops

#endif
