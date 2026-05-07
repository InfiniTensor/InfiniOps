#ifndef INFINI_OPS_BASE_SLOW_CONV3D_FORWARD_OUTPUT_H_
#define INFINI_OPS_BASE_SLOW_CONV3D_FORWARD_OUTPUT_H_

#include "operator.h"

namespace infini::ops {

class SlowConv3dForwardOutput : public Operator<SlowConv3dForwardOutput> {
 public:
  SlowConv3dForwardOutput(const Tensor self, const Tensor weight,
                          const std::vector<int64_t> kernel_size,
                          const std::vector<int64_t> stride,
                          const std::vector<int64_t> padding, Tensor output)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        weight_shape_{weight.shape()},
        weight_strides_{weight.strides()},
        weight_type_{weight.dtype()},
        output_shape_{output.shape()},
        output_strides_{output.strides()},
        output_type_{output.dtype()},
        device_index_{output.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor weight,
                          const std::vector<int64_t> kernel_size,
                          const std::vector<int64_t> stride,
                          const std::vector<int64_t> padding,
                          Tensor output) const = 0;

 protected:
  Tensor::Shape self_shape_;

  Tensor::Strides self_strides_;

  DataType self_type_;

  Tensor::Shape weight_shape_;

  Tensor::Strides weight_strides_;

  DataType weight_type_;

  Tensor::Shape output_shape_;

  Tensor::Strides output_strides_;

  DataType output_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
