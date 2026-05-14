#ifndef INFINI_OPS_BASE_SLOW_CONV3D_H_
#define INFINI_OPS_BASE_SLOW_CONV3D_H_

#include "operator.h"

namespace infini::ops {

class SlowConv3d : public Operator<SlowConv3d> {
 public:
  SlowConv3d(const Tensor input, const Tensor weight, const Tensor bias,
             const std::vector<int64_t> kernel_size,
             const std::vector<int64_t> stride,
             const std::vector<int64_t> padding, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        weight_shape_{weight.shape()},
        weight_strides_{weight.strides()},
        weight_type_{weight.dtype()},
        bias_shape_{bias.shape()},
        bias_strides_{bias.strides()},
        bias_type_{bias.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        kernel_size_{kernel_size},
        stride_{stride},
        padding_{padding},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor weight,
                          const Tensor bias,
                          const std::vector<int64_t> kernel_size,
                          const std::vector<int64_t> stride,
                          const std::vector<int64_t> padding,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape weight_shape_;

  Tensor::Strides weight_strides_;

  DataType weight_type_;

  Tensor::Shape bias_shape_;

  Tensor::Strides bias_strides_;

  DataType bias_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::vector<int64_t> kernel_size_{};

  std::vector<int64_t> stride_{};

  std::vector<int64_t> padding_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
