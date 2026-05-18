#ifndef INFINI_OPS_BASE_NLL_LOSS2D_H_
#define INFINI_OPS_BASE_NLL_LOSS2D_H_

#include "operator.h"

namespace infini::ops {

class NllLoss2d : public Operator<NllLoss2d> {
 public:
  NllLoss2d(const Tensor input, const Tensor target, const int64_t reduction,
            const int64_t ignore_index, Tensor out)
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
        ignore_index_{ignore_index},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor target,
                          const int64_t reduction, const int64_t ignore_index,
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

  int64_t ignore_index_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
