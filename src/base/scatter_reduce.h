#ifndef INFINI_OPS_BASE_SCATTER_REDUCE_H_
#define INFINI_OPS_BASE_SCATTER_REDUCE_H_

#include "operator.h"

namespace infini::ops {

class ScatterReduce : public Operator<ScatterReduce> {
 public:
  ScatterReduce(const Tensor self, const int64_t dim, const Tensor index,
                const Tensor src, const std::string reduce, Tensor out)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        index_shape_{index.shape()},
        index_strides_{index.strides()},
        index_type_{index.dtype()},
        src_shape_{src.shape()},
        src_strides_{src.strides()},
        src_type_{src.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor self, const int64_t dim,
                          const Tensor index, const Tensor src,
                          const std::string reduce, Tensor out) const = 0;

 protected:
  Tensor::Shape self_shape_;

  Tensor::Strides self_strides_;

  DataType self_type_;

  Tensor::Shape index_shape_;

  Tensor::Strides index_strides_;

  DataType index_type_;

  Tensor::Shape src_shape_;

  Tensor::Strides src_strides_;

  DataType src_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
