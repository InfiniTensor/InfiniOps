#ifndef INFINI_OPS_BASE_LINEAR_H_
#define INFINI_OPS_BASE_LINEAR_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

// Fused linear projection: out = a @ b (+ bias).
//
// When bias is present, computes out = a @ b + bias in a single dispatch.
// When bias is absent, computes out = a @ b (equivalent to Matmul).
// `trans_a` / `trans_b`: If true, transpose the last two dims before
// multiplying.
class Linear : public Operator<Linear> {
 public:
  Linear(const Tensor a, const Tensor b, std::optional<Tensor> bias,
         bool trans_a, bool trans_b, Tensor out)
      : a_shape_{a.shape()},
        b_shape_{b.shape()},
        out_shape_{out.shape()},
        a_strides_{a.strides()},
        b_strides_{b.strides()},
        out_strides_{out.strides()},
        trans_a_{trans_a},
        trans_b_{trans_b},
        has_bias_{bias.has_value()} {
    assert(a.dtype() == b.dtype() &&
           "operator `Linear` requires a and b to have the same dtype");
    assert(a.dtype() == out.dtype() &&
           "operator `Linear` requires a and out to have the same dtype");
    if (has_bias_) {
      assert(bias->dtype() == out.dtype() &&
             "operator `Linear` requires bias and out to have the same dtype");
    }
  }

  virtual void operator()(const Tensor a, const Tensor b,
                          std::optional<Tensor> bias, bool trans_a,
                          bool trans_b, Tensor out) const = 0;

 protected:
  Tensor::Shape a_shape_;

  Tensor::Shape b_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides a_strides_;

  Tensor::Strides b_strides_;

  Tensor::Strides out_strides_;

  bool trans_a_{false};

  bool trans_b_{false};

  bool has_bias_{false};
};

}  // namespace infini::ops

#endif
