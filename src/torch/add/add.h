#ifndef INFINI_OPS_TORCH_ADD_H_
#define INFINI_OPS_TORCH_ADD_H_

#include "base/add.h"
#include "torch/add/registry.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<Add, kDev, 1> : public Add {
 public:
  Operator(const Tensor input, const Tensor other, Tensor out)
      : Add{input, other, out},
        device_index_{out.device().index()} {}

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    // Use base-class stored metadata (not parameter tensors, which may be
    // moved-from by the `call()` dispatch path).
    auto at_input = ToAtenTensor<kDev>(
        const_cast<void*>(input.data()), input_shape_, input_strides_,
        input_type_, device_index_);
    auto at_other = ToAtenTensor<kDev>(
        const_cast<void*>(other.data()), other_shape_, other_strides_,
        other_type_, device_index_);
    auto at_out = ToAtenTensor<kDev>(
        out.data(), out_shape_, out_strides_, out_type_, device_index_);

    at::add_out(at_out, at_input, at_other);
  }

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
