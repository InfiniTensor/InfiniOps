#ifndef INFINI_OPS_BASE_STACK_BACKWARD_H_
#define INFINI_OPS_BASE_STACK_BACKWARD_H_

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include "base/validation.h"
#include "operator.h"

namespace infini::ops {

class StackBackward : public Operator<StackBackward> {
 public:
  StackBackward(const Tensor grad_output, const int64_t dim,
                std::vector<Tensor> grad_inputs)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        dim_{validation::NormalizeDim(dim, grad_output.ndim())},
        grad_inputs_{std::move(grad_inputs)},
        device_index_{
            grad_inputs_.empty() ? 0 : grad_inputs_.front().device().index()} {
    assert(!grad_inputs_.empty() &&
           "`StackBackward` requires at least one output");
    assert(static_cast<int64_t>(grad_inputs_.size()) ==
               grad_output.size(dim_) &&
           "`StackBackward` output count must equal grad_output.size(dim)");

    for (std::size_t i = 0; i < grad_inputs_.size(); ++i) {
      validation::AssertSameDevice({grad_output, grad_inputs_[i]});
      assert(grad_inputs_[i].ndim() == grad_output.ndim() - 1 &&
             "`StackBackward` output rank must be grad_output rank - 1");
      assert(grad_inputs_[i].dtype() == grad_output_type_ &&
             "`StackBackward` dtype mismatch");

      for (Tensor::Size d = 0, s = 0; d < grad_output.ndim(); ++d) {
        if (d == dim_) continue;
        assert(grad_inputs_[i].size(s) == grad_output.size(d) &&
               "`StackBackward` shape mismatch");

        ++s;
      }
      grad_input_shapes_.emplace_back(grad_inputs_[i].shape().begin(),
                                      grad_inputs_[i].shape().end());

      grad_input_strides_.emplace_back(grad_inputs_[i].strides().begin(),
                                       grad_inputs_[i].strides().end());

      grad_input_types_.push_back(grad_inputs_[i].dtype());
    }
  }

  virtual void operator()(const Tensor grad_output, const int64_t dim,
                          std::vector<Tensor> grad_inputs) const = 0;

 protected:
  Tensor::Shape grad_output_shape_;

  Tensor::Strides grad_output_strides_;

  DataType grad_output_type_;

  int64_t dim_{0};

  std::vector<Tensor> grad_inputs_{};

  std::vector<Tensor::Shape> grad_input_shapes_;

  std::vector<Tensor::Strides> grad_input_strides_;

  std::vector<DataType> grad_input_types_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_STACK_BACKWARD_H_
