#ifndef INFINI_OPS_BASE_HUBER_LOSS_H_
#define INFINI_OPS_BASE_HUBER_LOSS_H_

#include "operator.h"

namespace infini::ops {

class HuberLoss : public Operator<HuberLoss> {
 public:
  HuberLoss(const Tensor input, const Tensor target, const int64_t reduction,
            const double delta, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        target_shape_{target.shape()},
        target_strides_{target.strides()},
        target_type_{target.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        reduction_{reduction},
        delta_{delta},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor target,
                          const int64_t reduction, const double delta,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape target_shape_;

  Tensor::Strides target_strides_;

  DataType target_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int64_t reduction_{};

  double delta_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
