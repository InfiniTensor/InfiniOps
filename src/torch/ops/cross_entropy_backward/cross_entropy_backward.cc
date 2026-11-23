#include "torch/ops/cross_entropy_backward/cross_entropy_backward.h"

#include <vector>

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<CrossEntropyBackward, kDev, 1>::Operator(
    const Tensor input, const Tensor target, const Tensor grad_output,
    std::optional<Tensor> weight, int64_t ignore_index, std::string reduction,
    double label_smoothing, Tensor grad_input)
    : CrossEntropyBackward{input,           target,       grad_output,
                           weight,          ignore_index, reduction,
                           label_smoothing, grad_input} {}

template <Device::Type kDev>
void Operator<CrossEntropyBackward, kDev, 1>::operator()(
    const Tensor input, const Tensor target, const Tensor grad_output,
    std::optional<Tensor> weight, int64_t ignore_index, std::string reduction,
    double label_smoothing, Tensor grad_input) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  auto x = ToAtenTensor<kDev>(const_cast<void*>(input.data()), input_shape_,
                              input_strides_, input_type_, device_index_);
  auto target_tensor =
      ToAtenTensor<kDev>(const_cast<void*>(target.data()), target_shape_,
                         target_strides_, target_type_, device_index_);
  auto upstream = ToAtenTensor<kDev>(const_cast<void*>(grad_output.data()),
                                     grad_output_shape_, grad_output_strides_,
                                     grad_output_type_, device_index_);
  auto out =
      ToAtenTensor<kDev>(grad_input.data(), grad_input_shape_,
                         grad_input_strides_, grad_input_type_, device_index_);
  // Accumulate low-precision derivatives and reduction denominators in FP32.
  const auto compute_type =
      x.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
  x = x.to(compute_type);
  if (upstream.scalar_type() == at::kHalf ||
      upstream.scalar_type() == at::kBFloat16) {
    upstream = upstream.to(at::kFloat);
  }
  const int64_t dim = x.dim() == 1 ? 0 : 1;
  const auto classes = x.size(dim);
  const bool probability_target = target_shape_ == input_shape_;
  std::vector<int64_t> weight_shape(x.dim(), 1);
  weight_shape[dim] = classes;
  auto class_weight = at::ones({classes}, x.options());
  if (weight) {
    class_weight =
        ToAtenTensor<kDev>(const_cast<void*>(weight->data()), weight->shape(),
                           weight->strides(), weight->dtype(), device_index_);
    if (class_weight.scalar_type() != at::kDouble) {
      class_weight = class_weight.to(compute_type);
    }
  }
  at::Tensor valid;
  at::Tensor denominator;
  at::Tensor distribution;
  if (probability_target) {
    if (target_tensor.scalar_type() != at::kDouble) {
      target_tensor = target_tensor.to(compute_type);
    }
    distribution =
        target_tensor * (1.0 - label_smoothing) + label_smoothing / classes;
    denominator = at::scalar_tensor(x.numel() / classes, x.options());
  } else {
    auto indices = target_tensor.to(at::kLong);
    valid = indices.ne(ignore_index);
    auto safe_indices = indices.masked_fill(~valid, 0);
    distribution = at::full_like(x, label_smoothing / classes);
    auto selected = distribution.gather(dim, safe_indices.unsqueeze(dim));
    distribution.scatter_(dim, safe_indices.unsqueeze(dim),
                          selected + (1.0 - label_smoothing));
    const auto flat_indices = safe_indices.reshape({-1});
    const auto flat_weights = class_weight.index_select(0, flat_indices);
    auto selected_weight = flat_weights.reshape(indices.sizes());
    denominator = selected_weight.masked_fill(~valid, 0).sum();
  }
  auto weighted = distribution * class_weight.reshape(weight_shape);
  auto gradient = at::softmax(x, dim) * weighted.sum(dim, true) - weighted;
  if (reduction == "mean") gradient = gradient / denominator;
  if (reduction == "none") upstream = upstream.unsqueeze(dim);
  gradient = gradient * upstream;
  // Ignored positions stay zero even when a mean has zero total weight.
  if (!probability_target) gradient.masked_fill_(~valid.unsqueeze(dim), 0);
  out.copy_(gradient);
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(CrossEntropyBackward, 1)

}  // namespace infini::ops
