#ifndef INFINI_OPS_BASE_INDEX_FILL_H_
#define INFINI_OPS_BASE_INDEX_FILL_H_

#include "operator.h"

namespace infini::ops {

class IndexFill : public Operator<IndexFill> {
 public:
  IndexFill(Tensor input, const int64_t dim, const Tensor index,
            const double value)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        index_shape_{index.shape()},
        index_strides_{index.strides()},
        index_type_{index.dtype()},
        dim_{dim},
        value_{value},
        device_index_{input.device().index()} {}

  IndexFill(Tensor input, const int64_t dim, const Tensor index,
            const Tensor value)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        index_shape_{index.shape()},
        index_strides_{index.strides()},
        index_type_{index.dtype()},
        value_shape_{value.shape()},
        value_strides_{value.strides()},
        value_type_{value.dtype()},
        dim_{dim},
        device_index_{input.device().index()} {}

  virtual void operator()(Tensor input, const int64_t dim, const Tensor index,
                          const double value) const = 0;

  virtual void operator()(Tensor input, const int64_t dim, const Tensor index,
                          const Tensor value) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape index_shape_;

  Tensor::Strides index_strides_;

  DataType index_type_;

  int64_t dim_{};

  double value_{};

  Tensor::Shape value_shape_;

  Tensor::Strides value_strides_;

  DataType value_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
