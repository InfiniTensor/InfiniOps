#ifndef INFINI_OPS_BASE_TOPK_VALUES_H_
#define INFINI_OPS_BASE_TOPK_VALUES_H_

#include "operator.h"

namespace infini::ops {

class TopkValues : public Operator<TopkValues> {
 public:
  TopkValues(const Tensor self, const int64_t k, Tensor values, Tensor indices)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        values_shape_{values.shape()},
        values_strides_{values.strides()},
        values_type_{values.dtype()},
        indices_shape_{indices.shape()},
        indices_strides_{indices.strides()},
        indices_type_{indices.dtype()},
        device_index_{values.device().index()} {}

  virtual void operator()(const Tensor self, const int64_t k, Tensor values,
                          Tensor indices) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape values_shape_;
  Tensor::Strides values_strides_;
  DataType values_type_;
  Tensor::Shape indices_shape_;
  Tensor::Strides indices_strides_;
  DataType indices_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
