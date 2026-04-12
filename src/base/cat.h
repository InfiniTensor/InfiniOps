#ifndef INFINI_OPS_BASE_CAT_H_
#define INFINI_OPS_BASE_CAT_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "operator.h"

namespace infini::ops {

class Cat : public Operator<Cat> {
 public:
  Cat(const Tensor first_input, std::vector<Tensor> rest_inputs, int64_t dim,
      Tensor out)
      : dim_{static_cast<size_t>(dim >= 0 ? dim : dim + out.ndim())},
        input_count_{1 + rest_inputs.size()},
        dtype_{first_input.dtype()},
        ndim_{out.ndim()},
        output_size_{out.numel()} {
    assert(dim_ < ndim_ && "cat dim out of range");
    assert(out.dtype() == dtype_ &&
           "operator `Cat` requires all tensors to have the same dtype");

    for (const auto& t : rest_inputs) {
      assert(t.dtype() == dtype_ &&
             "operator `Cat` requires all tensors to have the same dtype");
      assert(t.ndim() == ndim_ &&
             "operator `Cat` requires all tensors to have the same ndim");
    }

    // Collect all input tensors.
    inputs_.reserve(input_count_);
    inputs_.push_back(first_input);

    for (auto& t : rest_inputs) {
      inputs_.push_back(std::move(t));
    }

    // Build cumulative sizes along the cat dimension.
    cum_dim_sizes_.resize(input_count_);
    cum_dim_sizes_[0] = inputs_[0].size(dim_);

    for (size_t i = 1; i < input_count_; ++i) {
      cum_dim_sizes_[i] = cum_dim_sizes_[i - 1] + inputs_[i].size(dim_);
    }

    // Compute outer_size (product of dims before cat dim) and inner_size
    // (product of dims after cat dim).
    outer_size_ = 1;

    for (size_t i = 0; i < dim_; ++i) {
      outer_size_ *= out.size(i);
    }

    inner_size_ = 1;

    for (size_t i = dim_ + 1; i < ndim_; ++i) {
      inner_size_ *= out.size(i);
    }
  }

  virtual void operator()(const Tensor first_input,
                           std::vector<Tensor> rest_inputs, int64_t dim,
                           Tensor out) const = 0;

 protected:
  size_t dim_{0};

  size_t input_count_{0};

  const DataType dtype_;

  size_t ndim_{0};

  size_t output_size_{0};

  size_t outer_size_{1};

  size_t inner_size_{1};

  std::vector<Tensor> inputs_;

  std::vector<size_t> cum_dim_sizes_;
};

}  // namespace infini::ops

#endif
