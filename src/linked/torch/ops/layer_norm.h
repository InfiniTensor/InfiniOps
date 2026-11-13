#ifndef INFINI_OPS_LINKED_TORCH_OPS_LAYER_NORM_H_
#define INFINI_OPS_LINKED_TORCH_OPS_LAYER_NORM_H_

#include <optional>
#include <vector>

#include "base/layer_norm.h"
#include "torch/tensor_.h"

namespace infini::ops::linked::torch {

template <typename Backend>
class TorchLayerNorm : public ::infini::ops::LayerNorm {
 public:
  TorchLayerNorm(const Tensor input, const Tensor weight,
                 const std::optional<Tensor> bias, float eps, Tensor out)
      : ::infini::ops::LayerNorm{input, weight, bias, eps, out},
        input_shape_{input.shape()},
        input_strides_{input.strides()},
        weight_shape_{weight.shape()},
        weight_strides_{weight.strides()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        input_type_{input.dtype()},
        weight_type_{weight.dtype()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  using ::infini::ops::LayerNorm::operator();

  void operator()(const Tensor input, const Tensor weight,
                  const std::optional<Tensor> bias, float eps,
                  Tensor out) const override {
    const typename Backend::StreamGuard stream_guard{
        Backend::GetStreamFromExternal(stream_, device_index_)};
    auto at_input = ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(input.data()), input_shape_, input_strides_,
        input_type_, device_index_);
    auto at_out = ToAtenTensor<Backend::kDeviceType>(
        out.data(), out_shape_, out_strides_, out_type_, device_index_);
    auto at_weight = ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(weight.data()), weight_shape_, weight_strides_,
        weight_type_, device_index_);
    c10::optional<at::Tensor> at_bias;
    if (bias.has_value()) {
      at_bias = c10::optional<at::Tensor>{ToAtenTensor<Backend::kDeviceType>(
          const_cast<void*>(bias->data()), bias->shape(), bias->strides(),
          bias->dtype(), device_index_)};
    }

    // Inference LayerNorm: only the normalized output is
    // needed; the mean and rstd outputs are discarded.
    std::vector<int64_t> normalized_shape{at_input.sizes().back()};
    auto result =
        at::native_layer_norm(at_input, at::IntArrayRef(normalized_shape),
                              at_weight, at_bias, static_cast<double>(eps));
    at_out.copy_(std::get<0>(result));
  }

 private:
  Tensor::Shape input_shape_;
  Tensor::Strides input_strides_;
  Tensor::Shape weight_shape_;
  Tensor::Strides weight_strides_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType input_type_;
  DataType weight_type_;
  DataType out_type_;
  int device_index_{0};
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_OPS_LAYER_NORM_H_
