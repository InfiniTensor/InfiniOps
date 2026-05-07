#ifndef INFINI_OPS_BASE_MASKED_SELECT_H_
#define INFINI_OPS_BASE_MASKED_SELECT_H_

#include "operator.h"

namespace infini::ops {

class MaskedSelect : public Operator<MaskedSelect> {
 public:
  MaskedSelect(const Tensor self, const Tensor mask, Tensor out)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        mask_shape_{mask.shape()},
        mask_strides_{mask.strides()},
        mask_type_{mask.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor mask,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape mask_shape_;
  Tensor::Strides mask_strides_;
  DataType mask_type_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType out_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
