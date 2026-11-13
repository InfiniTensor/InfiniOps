#ifndef INFINI_OPS_LINKED_TORCH_OPS_QUICK_GELU_H_
#define INFINI_OPS_LINKED_TORCH_OPS_QUICK_GELU_H_

#include "base/quick_gelu.h"
#include "torch/tensor_.h"

namespace infini::ops::linked::torch {

template <typename Backend>
class TorchQuickGelu : public ::infini::ops::QuickGelu {
 public:
  TorchQuickGelu(const Tensor input, Tensor out)
      : ::infini::ops::QuickGelu{input, out},
        input_shape_{input.shape()},
        input_strides_{input.strides()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        input_type_{input.dtype()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  using ::infini::ops::QuickGelu::operator();

  void operator()(const Tensor input, Tensor out) const override {
    const typename Backend::StreamGuard stream_guard{
        Backend::GetStreamFromExternal(stream_, device_index_)};
    auto at_input = ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(input.data()), input_shape_, input_strides_,
        input_type_, device_index_);
    auto at_out = ToAtenTensor<Backend::kDeviceType>(
        out.data(), out_shape_, out_strides_, out_type_, device_index_);

    auto backend_input =
        is_input_contiguous_ ? at_input : at_input.contiguous();
    auto backend_out = is_out_contiguous_
                           ? at_out
                           : at::empty(at_out.sizes(), at_out.options());

    Backend::Call(backend_out, backend_input);
    if (!is_out_contiguous_) {
      at_out.copy_(backend_out);
    }
  }

 private:
  Tensor::Shape input_shape_;
  Tensor::Strides input_strides_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType input_type_;
  DataType out_type_;
  bool is_input_contiguous_{false};
  bool is_out_contiguous_{false};
  int device_index_{0};
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_OPS_QUICK_GELU_H_
