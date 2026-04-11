#ifndef INFINI_OPS_BASE_LINEAR_H_
#define INFINI_OPS_BASE_LINEAR_H_

#include <algorithm>
#include <optional>

#include "operator.h"

namespace infini::ops {

class Linear : public Operator<Linear> {
 public:
  Linear(const Tensor a, const Tensor b, std::optional<Tensor> bias,
         bool trans_a, bool trans_b, Tensor out)
      : has_bias_{bias.has_value()},
        trans_a_{trans_a},
        trans_b_{trans_b},
        m_{out.size(-2)},
        n_{out.size(-1)},
        k_{trans_a_ ? a.size(-2) : a.size(-1)},
        a_type_{a.dtype()},
        b_type_{b.dtype()},
        out_type_{out.dtype()},
        a_strides_{a.strides()},
        b_strides_{b.strides()},
        out_strides_{out.strides()},
        lda_{std::max(a.stride(-2), a.stride(-1))},
        ldb_{std::max(b.stride(-2), b.stride(-1))},
        ldc_{std::max(out.stride(-2), out.stride(-1))},
        batch_count_{out.strides().size() > 2 ? out.size(-3) : 1},
        batch_stride_a_{a.strides().size() > 2 ? a.stride(-3) : 0},
        batch_stride_b_{b.strides().size() > 2 ? b.stride(-3) : 0},
        batch_stride_c_{out.strides().size() > 2 ? out.stride(-3) : 0} {
    // TODO: Check constraints.
  }

  virtual void operator()(const Tensor a, const Tensor b,
                          std::optional<Tensor> bias, bool trans_a,
                          bool trans_b, Tensor out) const = 0;

 protected:
  bool has_bias_{false};

  bool trans_a_{false};

  bool trans_b_{false};

  Tensor::Size m_{0};

  Tensor::Size n_{0};

  Tensor::Size k_{0};

  const DataType a_type_;

  const DataType b_type_;

  const DataType out_type_;

  Tensor::Strides a_strides_;

  Tensor::Strides b_strides_;

  Tensor::Strides out_strides_;

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
