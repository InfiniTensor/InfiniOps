#ifndef INFINI_OPS_BASE_MATMUL_H_
#define INFINI_OPS_BASE_MATMUL_H_

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class Matmul : public Operator<Matmul> {
 public:
  // `trans_a` / `trans_b`: If true, transpose the last two dims of `a` / `b`
  // before multiplying.  These are constructor parameters so the `CacheKey`
  // encodes the transposition and distinct descriptors are cached for each
  // combination.
  Matmul(const Tensor a, const Tensor b, Tensor c, bool trans_a, bool trans_b)
      : a_shape_{a.shape()},
        b_shape_{b.shape()},
        c_shape_{c.shape()},
        trans_a_{trans_a},
        trans_b_{trans_b} {
    assert(a.dtype() == b.dtype());
  }

  virtual void operator()(const Tensor a, const Tensor b, Tensor c,
                          bool trans_a, bool trans_b) const = 0;

 protected:
  Tensor::Shape a_shape_;

  Tensor::Shape b_shape_;

  Tensor::Shape c_shape_;

  bool trans_a_{false};

  bool trans_b_{false};
};

}  // namespace infini::ops

#endif
