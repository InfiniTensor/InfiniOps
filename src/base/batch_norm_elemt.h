#ifndef INFINI_OPS_BASE_BATCH_NORM_ELEMT_H_
#define INFINI_OPS_BASE_BATCH_NORM_ELEMT_H_

#include "operator.h"

namespace infini::ops {

class BatchNormElemt : public Operator<BatchNormElemt> {
 public:
  BatchNormElemt(const Tensor input, const Tensor mean, const Tensor invstd,
                 const double eps, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        mean_shape_{mean.shape()},
        mean_strides_{mean.strides()},
        mean_type_{mean.dtype()},
        invstd_shape_{invstd.shape()},
        invstd_strides_{invstd.strides()},
        invstd_type_{invstd.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor mean,
                          const Tensor invstd, const double eps,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;
  Tensor::Strides input_strides_;
  DataType input_type_;
  Tensor::Shape mean_shape_;
  Tensor::Strides mean_strides_;
  DataType mean_type_;
  Tensor::Shape invstd_shape_;
  Tensor::Strides invstd_strides_;
  DataType invstd_type_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType out_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
