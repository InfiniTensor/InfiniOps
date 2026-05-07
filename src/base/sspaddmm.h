#ifndef INFINI_OPS_BASE_SSPADDMM_H_
#define INFINI_OPS_BASE_SSPADDMM_H_

#include "operator.h"

namespace infini::ops {

class Sspaddmm : public Operator<Sspaddmm> {
 public:
  Sspaddmm(const Tensor self, const Tensor mat1, const Tensor mat2,
           const double beta, const double alpha, Tensor out)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        mat1_shape_{mat1.shape()},
        mat1_strides_{mat1.strides()},
        mat1_type_{mat1.dtype()},
        mat2_shape_{mat2.shape()},
        mat2_strides_{mat2.strides()},
        mat2_type_{mat2.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor mat1,
                          const Tensor mat2, const double beta,
                          const double alpha, Tensor out) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape mat1_shape_;
  Tensor::Strides mat1_strides_;
  DataType mat1_type_;
  Tensor::Shape mat2_shape_;
  Tensor::Strides mat2_strides_;
  DataType mat2_type_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType out_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
