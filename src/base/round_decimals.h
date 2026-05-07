#ifndef INFINI_OPS_BASE_ROUND_DECIMALS_H_
#define INFINI_OPS_BASE_ROUND_DECIMALS_H_

#include "operator.h"

namespace infini::ops {

class RoundDecimals : public Operator<RoundDecimals> {
 public:
  RoundDecimals(const Tensor self, const int64_t decimals, Tensor out)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor self, const int64_t decimals,
                          Tensor out) const = 0;

 protected:
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
