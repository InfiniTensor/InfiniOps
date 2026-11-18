#include "torch/ops/pow_backward/pow_backward.h"

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<PowBackward, kDev, 1>::Operator(const Tensor grad_output,
                                         const Tensor input,
                                         const double scalar,
                                         const bool scalar_is_base,
                                         Tensor grad_input)
    : PowBackward{grad_output, input, scalar, scalar_is_base, grad_input} {}

template <Device::Type kDev>
void Operator<PowBackward, kDev, 1>::operator()(const Tensor grad_output,
                                                const Tensor input,
                                                const double scalar,
                                                const bool scalar_is_base,
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

  if (!scalar_is_base_ && scalar_ == 0.0) {
    at_grad_input.zero_();
    return;
  }
  auto at_compute_input = at_input;
  auto at_compute_grad_output = at_grad_output;
  if (at_input.scalar_type() == at::kHalf ||
      at_input.scalar_type() == at::kBFloat16) {
    at_compute_input = at_input.to(at::kFloat);
  }
  if (at_grad_output.scalar_type() == at::kHalf ||
      at_grad_output.scalar_type() == at::kBFloat16) {
    at_compute_grad_output = at_grad_output.to(at::kFloat);
  }

  if (scalar_is_base_) {
    auto at_local_grad = at::pow(scalar_, at_compute_input) * std::log(scalar_);
    at_grad_input.copy_(at_compute_grad_output * at_local_grad);
  } else {
    auto at_local_grad = scalar_ * at::pow(at_compute_input, scalar_ - 1.0);
    at_grad_input.copy_(at_compute_grad_output * at_local_grad);
  }
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(PowBackward, 1)

}  // namespace infini::ops
