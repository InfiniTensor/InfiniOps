#ifndef INFINI_OPS_BASE_MAX_POOL3D_WITH_INDICES_H_
#define INFINI_OPS_BASE_MAX_POOL3D_WITH_INDICES_H_

#include "operator.h"

namespace infini::ops {

class MaxPool3dWithIndices : public Operator<MaxPool3dWithIndices> {
 public:
  MaxPool3dWithIndices(const Tensor self,
                       const std::vector<int64_t> kernel_size, Tensor out,
                       Tensor indices)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        indices_shape_{indices.shape()},
        indices_strides_{indices.strides()},
        indices_type_{indices.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor self,
                          const std::vector<int64_t> kernel_size, Tensor out,
                          Tensor indices) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType out_type_;
  Tensor::Shape indices_shape_;
  Tensor::Strides indices_strides_;
  DataType indices_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
