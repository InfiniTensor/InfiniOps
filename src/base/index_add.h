#ifndef INFINI_OPS_BASE_INDEX_ADD_H_
#define INFINI_OPS_BASE_INDEX_ADD_H_

#include "operator.h"

namespace infini::ops {

class IndexAdd : public Operator<IndexAdd> {
 public:
  IndexAdd(const Tensor self, const int64_t dim, const Tensor index,
           const Tensor source, const double alpha, Tensor out)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        index_shape_{index.shape()},
        index_strides_{index.strides()},
        index_type_{index.dtype()},
        source_shape_{source.shape()},
        source_strides_{source.strides()},
        source_type_{source.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor self, const int64_t dim,
                          const Tensor index, const Tensor source,
                          const double alpha, Tensor out) const = 0;

 protected:
  Tensor::Shape self_shape_;

  Tensor::Strides self_strides_;

  DataType self_type_;

  Tensor::Shape index_shape_;

  Tensor::Strides index_strides_;

  DataType index_type_;

  Tensor::Shape source_shape_;

  Tensor::Strides source_strides_;

  DataType source_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
