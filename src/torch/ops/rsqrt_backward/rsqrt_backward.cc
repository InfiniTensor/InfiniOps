#include "torch/ops/rsqrt_backward/rsqrt_backward.h"

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<RsqrtBackward, kDev, 1>::Operator(const Tensor grad_output,
                                           const Tensor input,
                                           Tensor grad_input)
    : RsqrtBackward{grad_output, input, grad_input} {}

template <Device::Type kDev>
void Operator<RsqrtBackward, kDev, 1>::operator()(const Tensor grad_output,
                                                  const Tensor input,
                                                  Tensor grad_input) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  auto at_grad_output = ToAtenTensor<kDev>(
      const_cast<void*>(grad_output.data()), grad_output_shape_,
      grad_output_strides_, grad_output_type_, device_index_);
  auto at_input =
      ToAtenTensor<kDev>(const_cast<void*>(input.data()), input_shape_,
                         input_strides_, input_type_, device_index_);
  auto at_grad_input =
      ToAtenTensor<kDev>(grad_input.data(), grad_input_shape_,
                         grad_input_strides_, grad_input_type_, device_index_);

  const auto compute_type =
      at_input.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
  const auto input_value = at_input.to(compute_type);
  const auto upstream = at_grad_output.to(compute_type);
  // Avoid cubing rsqrt before a small upstream gradient can scale it down.
  const auto result = upstream * (-0.5) * at::rsqrt(input_value) / input_value;
  at_grad_input.copy_(result);
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(RsqrtBackward, 1)

}  // namespace infini::ops
