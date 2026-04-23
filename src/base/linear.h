#ifndef INFINI_OPS_BASE_LINEAR_H_
#define INFINI_OPS_BASE_LINEAR_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

// Fused linear projection.  Primary form `(input, weight, bias?, out)`
// matches `F.linear(input, weight, bias)`: `weight` is pre-transposed as
// `[out_features, in_features]`, kernel computes `input @ weight^T`.  The
// 6-arg `(a, b, bias, trans_a, trans_b, out)` form is kept deprecated for
// callers that need explicit transpose flags.
class Linear : public Operator<Linear> {
 public:
  // Deprecated — use `(input, weight, bias, out)` instead.
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

  Linear(const Tensor input, const Tensor weight, std::optional<Tensor> bias,
         Tensor out)
      : Linear{input, weight, bias, /*trans_a=*/false, /*trans_b=*/true, out} {
    assert(weight.ndim() >= 2 &&
           "`Linear`: `weight` must have at least 2 dims "
           "`[..., out_features, in_features]`.");
    assert(weight.size(-1) == input.size(-1) &&
           "`Linear`: `weight.shape[-1]` must equal `input.shape[-1]` "
           "(`in_features`).");
    assert(weight.size(-2) == out.size(-1) &&
           "`Linear`: `weight.shape[-2]` must equal `out.shape[-1]` "
           "(`out_features`).");
  }

  // Deprecated — use `(input, weight, bias, out)` overload.
  virtual void operator()(const Tensor a, const Tensor b,
                          std::optional<Tensor> bias, bool trans_a,
                          bool trans_b, Tensor out) const = 0;

  virtual void operator()(const Tensor input, const Tensor weight,
                          std::optional<Tensor> bias, Tensor out) const {
    return operator()(input, weight, bias, /*trans_a=*/false,
                      /*trans_b=*/true, out);
  }

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
