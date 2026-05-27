#ifndef INFINI_OPS_BASE_SET_H_
#define INFINI_OPS_BASE_SET_H_

#include <vector>

#include "operator.h"

namespace infini::ops {

class Set : public Operator<Set> {
 public:
  Set(Tensor input, const Tensor source, const int64_t storage_offset,
      const std::vector<int64_t> size, const std::vector<int64_t> stride)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        source_shape_{source.shape()},
        source_strides_{source.strides()},
        source_type_{source.dtype()},
        storage_offset_{storage_offset},
        size_{size},
        stride_{stride},
        device_index_{input.device().index()} {}

  Set(Tensor input, const Tensor source)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        source_shape_{source.shape()},
        source_strides_{source.strides()},
        source_type_{source.dtype()},
        device_index_{input.device().index()} {}

  virtual void operator()(Tensor input, const Tensor source,
                          const int64_t storage_offset,
                          const std::vector<int64_t> size,
                          const std::vector<int64_t> stride) const = 0;

  virtual void operator()(Tensor input, const Tensor source) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape source_shape_;

  Tensor::Strides source_strides_;

  DataType source_type_;

  int64_t storage_offset_{};

  std::vector<int64_t> size_{};

  std::vector<int64_t> stride_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
