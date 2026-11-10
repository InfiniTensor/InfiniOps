#ifndef INFINI_OPS_LINKED_TORCH_OPS_PAD_H_
#define INFINI_OPS_LINKED_TORCH_OPS_PAD_H_

#include <optional>
#include <vector>

#include "base/pad.h"
#include "torch/tensor_.h"

namespace infini::ops::linked::torch {

template <typename Backend>
class TorchPad : public ::infini::ops::Pad {
 public:
  TorchPad(const Tensor input, const std::vector<int64_t> pad,
           const std::string mode, const std::optional<double> value,
           Tensor out)
      : ::infini::ops::Pad{input, pad, mode, value, out},
        input_shape_{input.shape()},
        input_strides_{input.strides()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        input_type_{input.dtype()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  using ::infini::ops::Pad::operator();

  void operator()(const Tensor input, const std::vector<int64_t> pad,
                  const std::string mode, const std::optional<double> value,
                  Tensor out) const override {
    const typename Backend::StreamGuard stream_guard{
        Backend::GetStreamFromExternal(stream_, device_index_)};
    auto at_input = ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(input.data()), input_shape_, input_strides_,
        input_type_, device_index_);
    auto at_out = ToAtenTensor<Backend::kDeviceType>(
        out.data(), out_shape_, out_strides_, out_type_, device_index_);
    (void)mode;
    Backend::Call(std::move(at_out), std::move(at_input), pad,
                  value.value_or(0.0));
  }

 private:
  Tensor::Shape input_shape_;
  Tensor::Strides input_strides_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType input_type_;
  DataType out_type_;
  int device_index_{0};
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_OPS_PAD_H_
