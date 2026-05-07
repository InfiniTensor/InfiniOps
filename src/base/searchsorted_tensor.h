#ifndef INFINI_OPS_BASE_SEARCHSORTED_TENSOR_H_
#define INFINI_OPS_BASE_SEARCHSORTED_TENSOR_H_

#include "operator.h"

namespace infini::ops {

class SearchsortedTensor : public Operator<SearchsortedTensor> {
 public:
  SearchsortedTensor(const Tensor sorted_sequence, const Tensor self,
                     Tensor out)
      : sorted_sequence_shape_{sorted_sequence.shape()},
        sorted_sequence_strides_{sorted_sequence.strides()},
        sorted_sequence_type_{sorted_sequence.dtype()},
        self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor sorted_sequence, const Tensor self,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape sorted_sequence_shape_;
  Tensor::Strides sorted_sequence_strides_;
  DataType sorted_sequence_type_;
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType out_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
