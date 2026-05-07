#ifndef INFINI_OPS_BASE_LINSPACE_TENSOR_SCALAR_H_
#define INFINI_OPS_BASE_LINSPACE_TENSOR_SCALAR_H_

#include "operator.h"

namespace infini::ops {

class LinspaceTensorScalar : public Operator<LinspaceTensorScalar> {
 public:
  LinspaceTensorScalar(const Tensor start, const double end,
                       const int64_t steps, Tensor out)
      : start_shape_{start.shape()},
        start_strides_{start.strides()},
        start_type_{start.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor start, const double end,
                          const int64_t steps, Tensor out) const = 0;

 protected:
  Tensor::Shape start_shape_;
  Tensor::Strides start_strides_;
  DataType start_type_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType out_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
