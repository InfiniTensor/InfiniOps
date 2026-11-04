#include "torch/ops/accumulate_grad/accumulate_grad.h"

#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<AccumulateGrad, kDev, 1>::Operator(const Tensor gradient,
                                            const double rate, Tensor tensor)
    : AccumulateGrad{gradient, rate, tensor},
      device_index_{tensor.device().index()} {}

template <Device::Type kDev>
void Operator<AccumulateGrad, kDev, 1>::operator()(const Tensor gradient,
                                                   const double rate,
                                                   Tensor tensor) const {
  auto at_gradient =
      ToAtenTensor<kDev>(const_cast<void*>(gradient.data()), gradient_shape_,
                         gradient_strides_, gradient_type_, device_index_);
  auto at_tensor = ToAtenTensor<kDev>(tensor.data(), tensor_shape_,
                                      tensor_strides_, tensor_type_,
                                      device_index_);

  const auto at_rate = at_tensor.is_floating_point()
                           ? at::Scalar(rate)
                           : at::Scalar(static_cast<int64_t>(rate));
  at_tensor.add_(at_gradient, at_rate);
}

template class Operator<AccumulateGrad, Device::Type::kCpu, 1>;
template class Operator<AccumulateGrad, Device::Type::kNvidia, 1>;
template class Operator<AccumulateGrad, Device::Type::kCambricon, 1>;
template class Operator<AccumulateGrad, Device::Type::kAscend, 1>;
template class Operator<AccumulateGrad, Device::Type::kMetax, 1>;
template class Operator<AccumulateGrad, Device::Type::kMoore, 1>;
template class Operator<AccumulateGrad, Device::Type::kIluvatar, 1>;
template class Operator<AccumulateGrad, Device::Type::kHygon, 1>;
template class Operator<AccumulateGrad, Device::Type::kMars, 1>;

}  // namespace infini::ops