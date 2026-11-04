#ifndef INFINI_OPS_BASE_MEAN_BACKWARD_H_
#define INFINI_OPS_BASE_MEAN_BACKWARD_H_

#include <cassert>
#include <cstdint>

#include "operator.h"

namespace infini::ops {

class MeanBackward : public Operator<MeanBackward> {
 public:
  MeanBackward(const Tensor grad_output, const int64_t dim, const bool keep_dim,
               Tensor grad_input)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        dim_{NormalizeDim(dim, grad_input.ndim())},
        keep_dim_{keep_dim},
        device_index_{grad_input.device().index()} {
    assert(grad_output_type_ == grad_input_type_ &&
           "`MeanBackward` requires matching dtypes");
    assert(grad_input.ndim() >= 1 && "`MeanBackward` requires ndim >= 1");
    if (keep_dim_) {
      assert(grad_output.ndim() == grad_input.ndim() &&
             grad_output.size(dim_) == 1 &&
             "`MeanBackward` keep_dim=true requires size-1 dim in grad_output");
      for (Tensor::Size i = 0; i < grad_input.ndim(); ++i) {
        if (i == dim_) continue;
        assert(grad_output.size(i) == grad_input.size(i) && "`MeanBackward` shape mismatch");
      }
    } else {
      assert(grad_output.ndim() == grad_input.ndim() - 1 &&
             "`MeanBackward` keep_dim=false requires rank-1 grad_output");
      for (Tensor::Size i = 0, s = 0; i < grad_input.ndim(); ++i) {
        if (i == dim_) continue;
        assert(grad_output.size(s) == grad_input.size(i) && "`MeanBackward` shape mismatch");
        ++s;
      }
    }
  }

  virtual void operator()(const Tensor grad_output, const int64_t dim,
                          const bool keep_dim, Tensor grad_input) const = 0;

 protected:
  static int64_t NormalizeDim(const int64_t dim, const Tensor::Size ndim) {
    assert(ndim > 0 && "`MeanBackward` requires ndim > 0");
    if (ndim == 0) return 0;
    const auto d = dim % static_cast<int64_t>(ndim);
    return d < 0 ? d + static_cast<int64_t>(ndim) : d;
  }

  Tensor::Shape grad_output_shape_;
  Tensor::Strides grad_output_strides_;
  DataType grad_output_type_;
  Tensor::Shape grad_input_shape_;
  Tensor::Strides grad_input_strides_;
  DataType grad_input_type_;
  int64_t dim_{0};
  bool keep_dim_{false};
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_MEAN_BACKWARD_H_
