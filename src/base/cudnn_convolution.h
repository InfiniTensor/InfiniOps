#ifndef INFINI_OPS_BASE_CUDNN_CONVOLUTION_H_
#define INFINI_OPS_BASE_CUDNN_CONVOLUTION_H_

#include "operator.h"

namespace infini::ops {

class CudnnConvolution : public Operator<CudnnConvolution> {
 public:
  CudnnConvolution(const Tensor self, const Tensor weight,
                   const std::vector<int64_t> padding,
                   const std::vector<int64_t> stride,
                   const std::vector<int64_t> dilation, const int64_t groups,
                   const bool benchmark, const bool deterministic,
                   const bool allow_tf32, Tensor out)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        weight_shape_{weight.shape()},
        weight_strides_{weight.strides()},
        weight_type_{weight.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor weight,
                          const std::vector<int64_t> padding,
                          const std::vector<int64_t> stride,
                          const std::vector<int64_t> dilation,
                          const int64_t groups, const bool benchmark,
                          const bool deterministic, const bool allow_tf32,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape self_shape_;

  Tensor::Strides self_strides_;

  DataType self_type_;

  Tensor::Shape weight_shape_;

  Tensor::Strides weight_strides_;

  DataType weight_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
