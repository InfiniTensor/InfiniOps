#include "torch/ops/rsqrt_backward/rsqrt_backward.h"

#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<RsqrtBackward, kDev, 1>::Operator(const Tensor grad_output,
                                           const Tensor input, Tensor grad_input)
    : RsqrtBackward{grad_output, input, grad_input},
      device_index_{grad_input.device().index()} {}

template <Device::Type kDev>
void Operator<RsqrtBackward, kDev, 1>::operator()(const Tensor grad_output,
                                                  const Tensor input,
                                                  Tensor grad_input) const {
  auto at_grad_output =
      ToAtenTensor<kDev>(const_cast<void*>(grad_output.data()),
                         grad_output_shape_, grad_output_strides_,
                         grad_output_type_, device_index_);
  auto at_input = ToAtenTensor<kDev>(const_cast<void*>(input.data()),
                                     input_shape_, input_strides_, input_type_,
                                     device_index_);
  auto at_grad_input =
      ToAtenTensor<kDev>(grad_input.data(), grad_input_shape_,
                         grad_input_strides_, grad_input_type_, device_index_);

  at_grad_input.copy_(at_grad_output * (-0.5) * at::pow(at::rsqrt(at_input), 3));
}

template class Operator<RsqrtBackward, Device::Type::kCpu, 1>;
template class Operator<RsqrtBackward, Device::Type::kNvidia, 1>;
template class Operator<RsqrtBackward, Device::Type::kCambricon, 1>;
template class Operator<RsqrtBackward, Device::Type::kAscend, 1>;
template class Operator<RsqrtBackward, Device::Type::kMetax, 1>;
template class Operator<RsqrtBackward, Device::Type::kMoore, 1>;
template class Operator<RsqrtBackward, Device::Type::kIluvatar, 1>;
template class Operator<RsqrtBackward, Device::Type::kHygon, 1>;
template class Operator<RsqrtBackward, Device::Type::kMars, 1>;

}  // namespace infini::ops