#ifndef INFINI_OPS_BASE_SEARCHSORTED_H_
#define INFINI_OPS_BASE_SEARCHSORTED_H_

#include "operator.h"

namespace infini::ops {

class Searchsorted : public Operator<Searchsorted> {
 public:
  Searchsorted(const Tensor sorted_sequence, const Tensor input,
               const bool out_int32, const bool right, Tensor out)
      : sorted_sequence_shape_{sorted_sequence.shape()},
        sorted_sequence_strides_{sorted_sequence.strides()},
        sorted_sequence_type_{sorted_sequence.dtype()},
        input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        out_int32_{out_int32},
        right_{right},
        device_index_{out.device().index()} {}

  Searchsorted(const Tensor sorted_sequence, const double input,
               const bool out_int32, const bool right, Tensor out)
      : sorted_sequence_shape_{sorted_sequence.shape()},
        sorted_sequence_strides_{sorted_sequence.strides()},
        sorted_sequence_type_{sorted_sequence.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        out_int32_{out_int32},
        right_{right},
        input_{input},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor sorted_sequence, const Tensor input,
                          const bool out_int32, const bool right,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor sorted_sequence, const double input,
                          const bool out_int32, const bool right,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape sorted_sequence_shape_;

  Tensor::Strides sorted_sequence_strides_;

  DataType sorted_sequence_type_;

  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  bool out_int32_{};

  bool right_{};

  double input_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif