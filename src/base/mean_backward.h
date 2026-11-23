#ifndef INFINI_OPS_BASE_MEAN_BACKWARD_H_
#define INFINI_OPS_BASE_MEAN_BACKWARD_H_

#include <cassert>
#include <cstdint>

#include "base/validation.h"
#include "operator.h"

namespace infini::ops {

class MeanBackward : public Operator<MeanBackward> {
 public:
  MeanBackward(const Tensor grad_output, const int64_t dim, const bool keepdim,
               Tensor grad_input)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        dim_{validation::NormalizeDim(dim, grad_input.ndim())},
        keepdim_{keepdim},
        device_index_{grad_input.device().index()} {
    validation::AssertSameDevice({grad_output, grad_input});
    assert(grad_output_type_ == grad_input_type_ &&
           "`MeanBackward` requires matching dtypes");

    assert(grad_input.ndim() >= 1 && "`MeanBackward` requires ndim >= 1");
    if (keepdim_) {
      assert(grad_output.ndim() == grad_input.ndim() &&
             grad_output.size(dim_) == 1 &&
             "`MeanBackward` keepdim=true requires size-1 dim in grad_output");

      for (Tensor::Size i = 0; i < grad_input.ndim(); ++i) {
        if (i == dim_) continue;
        assert(grad_output.size(i) == grad_input.size(i) &&
               "`MeanBackward` shape mismatch");
      }
    } else {
      assert(grad_output.ndim() == grad_input.ndim() - 1 &&
             "`MeanBackward` keepdim=false requires rank-1 grad_output");

      for (Tensor::Size i = 0, s = 0; i < grad_input.ndim(); ++i) {
        if (i == dim_) continue;
        assert(grad_output.size(s) == grad_input.size(i) &&
               "`MeanBackward` shape mismatch");

        ++s;
      }
    }
  }

  virtual void operator()(const Tensor grad_output, const int64_t dim,
                          const bool keepdim, Tensor grad_input) const = 0;

 protected:
  Tensor::Shape grad_output_shape_;

  Tensor::Strides grad_output_strides_;

  DataType grad_output_type_;

  Tensor::Shape grad_input_shape_;

  Tensor::Strides grad_input_strides_;

  DataType grad_input_type_;

  int64_t dim_{0};

  bool keepdim_{false};

  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_MEAN_BACKWARD_H_
