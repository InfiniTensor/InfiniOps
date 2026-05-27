#ifndef INFINI_OPS_BASE_MASKED_FILL_H_
#define INFINI_OPS_BASE_MASKED_FILL_H_

#include "operator.h"

namespace infini::ops {

class MaskedFill : public Operator<MaskedFill> {
 public:
  MaskedFill(Tensor input, const Tensor mask, const double value)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        mask_shape_{mask.shape()},
        mask_strides_{mask.strides()},
        mask_type_{mask.dtype()},
        value_{value},
        device_index_{input.device().index()} {}

  MaskedFill(Tensor input, const Tensor mask, const Tensor value)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        mask_shape_{mask.shape()},
        mask_strides_{mask.strides()},
        mask_type_{mask.dtype()},
        value_shape_{value.shape()},
        value_strides_{value.strides()},
        value_type_{value.dtype()},
        device_index_{input.device().index()} {}

  virtual void operator()(Tensor input, const Tensor mask,
                          const double value) const = 0;

  virtual void operator()(Tensor input, const Tensor mask,
                          const Tensor value) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape mask_shape_;

  Tensor::Strides mask_strides_;

  DataType mask_type_;

  double value_{};

  Tensor::Shape value_shape_;

  Tensor::Strides value_strides_;

  DataType value_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
