#ifndef INFINI_OPS_BASE_ADD_H_
#define INFINI_OPS_BASE_ADD_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class Add : public Operator<Add> {
 public:
  Add(const Tensor a, const Tensor b, Tensor c)
      : ndim_{c.ndim()},
        output_size_{c.numel()},
        a_type_{a.dtype()},
        b_type_{b.dtype()},
        c_type_{c.dtype()},
        a_shape_{a.shape()},
        b_shape_{b.shape()},
        c_shape_{c.shape()},
        a_strides_{a.strides()},
        b_strides_{b.strides()},
        c_strides_{c.strides()},
        is_a_contiguous_{a.IsContiguous()},
        is_b_contiguous_{b.IsContiguous()},
        is_c_contiguous_{c.IsContiguous()} {
    assert(!c.HasBroadcastDim() &&
           "The output of `Add` should NOT have broadcasted dim!");
    // TODO(lzm): support mix-precision later using the generic elementwise
    // framework.
    assert(a_type_ == b_type_ && b_type_ == c_type_ &&
           "Operator `Add` requires all input and output Tensors to have the "
           "same dtype");
  }

  virtual void operator()(void* stream, const Tensor a, const Tensor b,
                          Tensor c) const = 0;

 protected:
  Tensor::Size ndim_{0};

  Tensor::Size output_size_{0};

  const DataType a_type_;

  const DataType b_type_;

  const DataType c_type_;

  Tensor::Shape a_shape_;

  Tensor::Shape b_shape_;

  Tensor::Shape c_shape_;

  Tensor::Strides a_strides_;

  Tensor::Strides b_strides_;

  Tensor::Strides c_strides_;

  bool is_a_contiguous_{false};

  bool is_b_contiguous_{false};

  bool is_c_contiguous_{false};
};

}  // namespace infini::ops

#endif
