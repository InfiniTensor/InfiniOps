#ifndef INFINI_OPS_BASE_STACK_BACKWARD_H_
#define INFINI_OPS_BASE_STACK_BACKWARD_H_

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

#include "operator.h"

namespace infini::ops {

class StackBackward : public Operator<StackBackward> {
 public:
  StackBackward(const Tensor grad_output, const int64_t dim,
                std::vector<Tensor> outs)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        dim_{NormalizeDim(dim, grad_output.ndim())},
        outs_{std::move(outs)},
        device_index_{outs_.empty() ? 0 : outs_.front().device().index()} {
    assert(!outs_.empty() && "`StackBackward` requires at least one output");
    assert(static_cast<int64_t>(outs_.size()) == grad_output.size(dim_) &&
           "`StackBackward` output count must equal grad_output.size(dim)");

    for (std::size_t i = 0; i < outs_.size(); ++i) {
      AssertTrainingDevices({grad_output, outs_[i]});
      assert(outs_[i].ndim() == grad_output.ndim() - 1 &&
             "`StackBackward` output rank must be grad_output rank - 1");
      assert(outs_[i].dtype() == grad_output_type_ &&
             "`StackBackward` dtype mismatch");
      for (Tensor::Size d = 0, s = 0; d < grad_output.ndim(); ++d) {
        if (d == dim_) continue;
        assert(outs_[i].size(s) == grad_output.size(d) &&
               "`StackBackward` shape mismatch");
        ++s;
      }
      out_shapes_.emplace_back(outs_[i].shape().begin(),
                               outs_[i].shape().end());

      out_strides_.emplace_back(outs_[i].strides().begin(),
                                outs_[i].strides().end());

      out_types_.push_back(outs_[i].dtype());
    }
  }

  virtual void operator()(const Tensor grad_output, const int64_t dim,
                          std::vector<Tensor> outs) const = 0;

 protected:
  static void AssertTrainingDevices(std::initializer_list<Tensor> tensors) {
    if (tensors.size() == 0) return;
    const auto device = tensors.begin()->device();
    for (const auto& tensor : tensors) {
      assert(tensor.device() == device &&
             "Training tensors must share a device type and index.");
    }
  }

  static int64_t NormalizeDim(const int64_t dim, const Tensor::Size ndim) {
    const auto rank = static_cast<int64_t>(ndim);
    assert(rank > 0 && "Training dimension requires a non-scalar tensor.");
    assert(dim >= -rank && dim < rank && "Training dimension is out of range.");
    return dim < 0 ? dim + rank : dim;
  }

  Tensor::Shape grad_output_shape_;
  Tensor::Strides grad_output_strides_;
  DataType grad_output_type_;
  int64_t dim_{0};
  std::vector<Tensor> outs_{};
  std::vector<Tensor::Shape> out_shapes_;
  std::vector<Tensor::Strides> out_strides_;
  std::vector<DataType> out_types_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_STACK_BACKWARD_H_
