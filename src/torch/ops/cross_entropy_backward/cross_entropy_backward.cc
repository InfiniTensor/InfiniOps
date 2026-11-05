#include "torch/ops/cross_entropy_backward/cross_entropy_backward.h"

#include <cassert>

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<CrossEntropyBackward, kDev, 1>::Operator(const Tensor input,
                                                  const Tensor target,
                                                  const Tensor grad_output,
                                                  Tensor grad_input)
    : CrossEntropyBackward{input, target, grad_output, grad_input} {}

template <Device::Type kDev>
void Operator<CrossEntropyBackward, kDev, 1>::operator()(
    const Tensor input, const Tensor target, const Tensor grad_output,
    Tensor grad_input) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  auto at_input =
      ToAtenTensor<kDev>(const_cast<void*>(input.data()), input_shape_,
                         input_strides_, input_type_, device_index_);
  auto at_target =
      ToAtenTensor<kDev>(const_cast<void*>(target.data()), target_shape_,
                         target_strides_, target_type_, device_index_);
  auto at_grad_output = ToAtenTensor<kDev>(
      const_cast<void*>(grad_output.data()), grad_output_shape_,
      grad_output_strides_, grad_output_type_, device_index_);
  auto at_grad_input =
      ToAtenTensor<kDev>(grad_input.data(), grad_input_shape_,
                         grad_input_strides_, grad_input_type_, device_index_);

  const auto classes = at_input.size(1);
  assert(classes > 0 &&
         "CrossEntropyBackward requires a non-empty class dimension");

  auto gradient = at::softmax(at_input, 1).to(at_grad_input.scalar_type());
  auto target_with_class_dim = at_target.to(at::kLong).unsqueeze(1);
  const auto valid_target =
      target_with_class_dim.ne(CrossEntropyBackward::kIgnoreIndex);
  const auto valid_range =
      (target_with_class_dim >= 0) & (target_with_class_dim < classes);
  assert(at::all((~valid_target) | valid_range).template item<bool>() &&
         "CrossEntropyBackward target is outside [0, classes) or ignore_index");

  auto safe_target = target_with_class_dim.masked_fill(~valid_target, 0);
  auto minus_one = -valid_target.to(gradient.scalar_type());
  gradient.scatter_add_(1, safe_target, minus_one);
  gradient.mul_(valid_target.to(gradient.scalar_type()));

  const auto valid_count =
      valid_target.sum().to(gradient.scalar_type()).clamp_min(1);
  gradient.mul_(at_grad_output.reshape({1}).to(gradient.scalar_type()) /
                valid_count);
  at_grad_input.copy_(gradient);
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(CrossEntropyBackward, 1)

}  // namespace infini::ops
