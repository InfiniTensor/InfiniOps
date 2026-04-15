#ifndef INFINI_OPS_BASE_CAT_H_
#define INFINI_OPS_BASE_CAT_H_

#include <vector>

#include "operator.h"

namespace infini::ops {

class Cat : public Operator<Cat> {
 public:
  Cat(const Tensor first_input, std::vector<Tensor> rest_inputs, int64_t dim,
      Tensor out)
      : input_count_{1 + rest_inputs.size()} {
    assert(input_count_ >= 2 && "Cat requires at least 2 input tensors");

    auto ndim = static_cast<int64_t>(out.ndim());
    // Normalize negative dim (e.g. -1 means last dimension).
    dim_ = dim < 0 ? dim + ndim : dim;
    assert(dim_ >= 0 && dim_ < ndim && "Cat dim out of range");
  }

  virtual void operator()(const Tensor first_input,
                          std::vector<Tensor> rest_inputs, int64_t dim,
                          Tensor out) const = 0;

 protected:
  int64_t dim_;

  size_t input_count_;
};

}  // namespace infini::ops

#endif
