#ifndef INFINI_OPS_BASE_NATIVE_BATCH_NORM_H_
#define INFINI_OPS_BASE_NATIVE_BATCH_NORM_H_

#include "operator.h"

namespace infini::ops {

class NativeBatchNorm : public Operator<NativeBatchNorm> {
 public:
  NativeBatchNorm(const Tensor input, const bool training,
                  const double momentum, const double eps, Tensor out,
                  Tensor save_mean, Tensor save_invstd)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        save_mean_shape_{save_mean.shape()},
        save_mean_strides_{save_mean.strides()},
        save_mean_type_{save_mean.dtype()},
        save_invstd_shape_{save_invstd.shape()},
        save_invstd_strides_{save_invstd.strides()},
        save_invstd_type_{save_invstd.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const bool training,
                          const double momentum, const double eps, Tensor out,
                          Tensor save_mean, Tensor save_invstd) const = 0;

 protected:
  Tensor::Shape input_shape_;
  Tensor::Strides input_strides_;
  DataType input_type_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType out_type_;
  Tensor::Shape save_mean_shape_;
  Tensor::Strides save_mean_strides_;
  DataType save_mean_type_;
  Tensor::Shape save_invstd_shape_;
  Tensor::Strides save_invstd_strides_;
  DataType save_invstd_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
