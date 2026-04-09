#ifndef INFINI_OPS_BASE_CAT_H_
#define INFINI_OPS_BASE_CAT_H_

#include <vector>

#include "operator.h"

namespace infini::ops {

class Cat : public Operator<Cat> {
 public:
  Cat(const Tensor first_input, std::vector<Tensor> rest_inputs, int64_t dim,
      Tensor out)
      : dim_{dim}, input_count_{1 + rest_inputs.size()} {
    assert(input_count_ >= 2 && "Cat requires at least 2 input tensors");

    auto ndim = out.ndim();
    assert(dim >= 0 && dim < static_cast<int64_t>(ndim) &&
           "Cat dim out of range");
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
