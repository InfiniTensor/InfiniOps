#ifndef INFINI_OPS_BASE_MAX_UNPOOL3D_H_
#define INFINI_OPS_BASE_MAX_UNPOOL3D_H_

#include "operator.h"

namespace infini::ops {

class MaxUnpool3d : public Operator<MaxUnpool3d> {
 public:
  MaxUnpool3d(const Tensor self, const Tensor indices,
              const std::vector<int64_t> output_size,
              const std::vector<int64_t> stride,
              const std::vector<int64_t> padding, Tensor out)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        indices_shape_{indices.shape()},
        indices_strides_{indices.strides()},
        indices_type_{indices.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor indices,
                          const std::vector<int64_t> output_size,
                          const std::vector<int64_t> stride,
                          const std::vector<int64_t> padding,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape indices_shape_;
  Tensor::Strides indices_strides_;
  DataType indices_type_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType out_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
