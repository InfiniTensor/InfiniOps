#ifndef INFINI_OPS_BASE_CAUSAL_SOFTMAX_H_
#define INFINI_OPS_BASE_CAUSAL_SOFTMAX_H_

#include <cassert>
#include <cstddef>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class CausalSoftmax : public Operator<CausalSoftmax> {
 public:
  CausalSoftmax(const Tensor y, const Tensor x)
      : dtype_{y.dtype()},
        ndim_{y.ndim()},
        batch_size_{ndim_ == 2 ? 1 : y.size(-3)},
        seq_len_{y.size(-2)},
        total_seq_len_{y.size(-1)},
        y_strides_{y.strides()},
        x_strides_{x.strides()} {
    assert(y.shape() == x.shape() &&
           "CausalSoftmax requires y and x same shape");
    assert(y.dtype() == x.dtype() &&
           "CausalSoftmax requires y and x same dtype");
    assert(ndim_ == 2 ||
           ndim_ == 3 && "CausalSoftmax requires 2D or 3D tensor");
    assert(seq_len_ <= total_seq_len_ &&
           "CausalSoftmax requires shape[-2] <= shape[-1]");
  }

  virtual void operator()(void* stream, Tensor y, const Tensor x) const = 0;

 protected:
  const DataType dtype_;

  Tensor::Size ndim_{0};

  Tensor::Size batch_size_{0};

  Tensor::Size seq_len_{0};

  Tensor::Size total_seq_len_{0};

  Tensor::Strides y_strides_;

  Tensor::Strides x_strides_;
};

}  // namespace infini::ops

#endif
