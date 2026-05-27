#ifndef INFINI_OPS_BASE_TENSORDOT_H_
#define INFINI_OPS_BASE_TENSORDOT_H_

#include <vector>

#include "operator.h"

namespace infini::ops {

class Tensordot : public Operator<Tensordot> {
 public:
  Tensordot(const Tensor input, const Tensor other,
            const std::vector<int64_t> dims_self,
            const std::vector<int64_t> dims_other, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        other_shape_{other.shape()},
        other_strides_{other.strides()},
        other_type_{other.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dims_self_{dims_self},
        dims_other_{dims_other},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor other,
                          const std::vector<int64_t> dims_self,
                          const std::vector<int64_t> dims_other,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape other_shape_;

  Tensor::Strides other_strides_;

  DataType other_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::vector<int64_t> dims_self_{};

  std::vector<int64_t> dims_other_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
