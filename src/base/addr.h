#ifndef INFINI_OPS_BASE_ADDR_H_
#define INFINI_OPS_BASE_ADDR_H_

#include "operator.h"

namespace infini::ops {

class Addr : public Operator<Addr> {
 public:
  Addr(const Tensor self, const Tensor vec1, const Tensor vec2,
       const double beta, const double alpha, Tensor out)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        vec1_shape_{vec1.shape()},
        vec1_strides_{vec1.strides()},
        vec1_type_{vec1.dtype()},
        vec2_shape_{vec2.shape()},
        vec2_strides_{vec2.strides()},
        vec2_type_{vec2.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor vec1,
                          const Tensor vec2, const double beta,
                          const double alpha, Tensor out) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape vec1_shape_;
  Tensor::Strides vec1_strides_;
  DataType vec1_type_;
  Tensor::Shape vec2_shape_;
  Tensor::Strides vec2_strides_;
  DataType vec2_type_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType out_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
