#ifndef INFINI_OPS_BASE_MATMUL_H_
#define INFINI_OPS_BASE_MATMUL_H_

#include <algorithm>

#include "operator.h"

namespace infini::ops {

class Matmul : public Operator<Matmul> {
 public:
  Matmul(const Tensor a, const Tensor b, Tensor c, bool trans_a, bool trans_b)
      : trans_a_{trans_a},
        trans_b_{trans_b},
        m_{c.size(-2)},
        n_{c.size(-1)},
        k_{trans_a_ ? a.size(-2) : a.size(-1)},
        a_type_{a.dtype()},
        b_type_{b.dtype()},
        c_type_{c.dtype()},
        a_strides_{a.strides()},
        b_strides_{b.strides()},
        c_strides_{c.strides()},
        lda_{std::max(a.stride(-2), a.stride(-1))},
        ldb_{std::max(b.stride(-2), b.stride(-1))},
        ldc_{std::max(c.stride(-2), c.stride(-1))},
        batch_count_{c.strides().size() > 2 ? c.size(-3) : 1},
        batch_stride_a_{a.strides().size() > 2 ? a.stride(-3) : 0},
        batch_stride_b_{b.strides().size() > 2 ? b.stride(-3) : 0},
        batch_stride_c_{c.strides().size() > 2 ? c.stride(-3) : 0} {
    // TODO: Check constraints.
  }

  Matmul(const Tensor a, const Tensor b, Tensor c)
      : Matmul{a, b, c, false, false} {}

  virtual void operator()(const Tensor a, const Tensor b, Tensor c,
                          bool trans_a, bool trans_b) const = 0;

  virtual void operator()(const Tensor a, const Tensor b, Tensor c) const {
    return operator()(a, b, c, false, false);
  }

 protected:
  bool trans_a_{false};

  bool trans_b_{false};

  Tensor::Size m_{0};

  Tensor::Size n_{0};

  Tensor::Size k_{0};

  const DataType a_type_;

  const DataType b_type_;

  const DataType c_type_;

  Tensor::Strides a_strides_;

  Tensor::Strides b_strides_;

  Tensor::Strides c_strides_;

  Tensor::Stride lda_{0};

  Tensor::Stride ldb_{0};

  Tensor::Stride ldc_{0};

  Tensor::Size batch_count_{1};

  Tensor::Stride batch_stride_a_{0};

  Tensor::Stride batch_stride_b_{0};

  Tensor::Stride batch_stride_c_{0};
};

}  // namespace infini::ops

#endif
