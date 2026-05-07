#ifndef INFINI_OPS_BASE_UPSAMPLE_NEAREST3D_BACKWARD_GRAD_INPUT_H_
#define INFINI_OPS_BASE_UPSAMPLE_NEAREST3D_BACKWARD_GRAD_INPUT_H_

#include "operator.h"

namespace infini::ops {

class UpsampleNearest3dBackwardGradInput
    : public Operator<UpsampleNearest3dBackwardGradInput> {
 public:
  UpsampleNearest3dBackwardGradInput(const Tensor grad_output,
                                     const std::vector<int64_t> output_size,
                                     const std::vector<int64_t> input_size,
                                     Tensor grad_input)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        device_index_{grad_input.device().index()} {}

  virtual void operator()(const Tensor grad_output,
                          const std::vector<int64_t> output_size,
                          const std::vector<int64_t> input_size,
                          Tensor grad_input) const = 0;

 protected:
  Tensor::Shape grad_output_shape_;

  Tensor::Strides grad_output_strides_;

  DataType grad_output_type_;

  Tensor::Shape grad_input_shape_;

  Tensor::Strides grad_input_strides_;

  DataType grad_input_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
