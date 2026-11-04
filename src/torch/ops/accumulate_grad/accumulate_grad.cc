#include "torch/ops/accumulate_grad/accumulate_grad.h"

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
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
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  auto at_gradient =
      ToAtenTensor<kDev>(const_cast<void*>(gradient.data()), gradient_shape_,
                         gradient_strides_, gradient_type_, device_index_);
  auto at_tensor =
      ToAtenTensor<kDev>(tensor.data(), tensor_shape_, tensor_strides_,
                         tensor_type_, device_index_);

  const auto at_rate = at_tensor.is_floating_point()
                           ? at::Scalar(rate)
                           : at::Scalar(static_cast<int64_t>(rate));
  at_tensor.add_(at_gradient, at_rate);
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(AccumulateGrad, 1)

}  // namespace infini::ops
