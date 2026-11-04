#include "torch/ops/adam_accumulate_grad/adam_accumulate_grad.h"

#include <cmath>

#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<AdamAccumulateGrad, kDev, 1>::Operator(
    const Tensor grad, Tensor param, Tensor m, Tensor v,
    const double learning_rate, const double beta1, const double beta2,
    const double eps, const int64_t step)
    : AdamAccumulateGrad{grad, param, m, v, learning_rate,
                         beta1, beta2, eps, step},
      device_index_{param.device().index()} {}

template <Device::Type kDev>
void Operator<AdamAccumulateGrad, kDev, 1>::operator()(
    const Tensor grad, Tensor param, Tensor m, Tensor v,
    const double learning_rate, const double beta1, const double beta2,
    const double eps, const int64_t step) const {
  auto at_grad = ToAtenTensor<kDev>(const_cast<void*>(grad.data()), grad_shape_,
                                    grad_strides_, grad_type_, device_index_);
  auto at_param = ToAtenTensor<kDev>(param.data(), param_shape_, param_strides_,
                                     param_type_, device_index_);
  auto at_m = ToAtenTensor<kDev>(m.data(), m_shape_, m_strides_, m_type_,
                                 device_index_);
  auto at_v = ToAtenTensor<kDev>(v.data(), v_shape_, v_strides_, v_type_,
                                 device_index_);

  at_m.mul_(beta1_).add_(at_grad, 1.0 - beta1_);
  at_v.mul_(beta2_).addcmul_(at_grad, at_grad, 1.0 - beta2_);

  const double bcm = 1.0 - std::pow(beta1_, static_cast<double>(step_));
  const double bcv = 1.0 - std::pow(beta2_, static_cast<double>(step_));
  const auto m_hat = at_m / bcm;
  const auto v_hat = at_v / bcv;

  at_param.add_(-learning_rate_ * m_hat / (at::sqrt(v_hat) + eps_));
}

template class Operator<AdamAccumulateGrad, Device::Type::kCpu, 1>;
template class Operator<AdamAccumulateGrad, Device::Type::kNvidia, 1>;
template class Operator<AdamAccumulateGrad, Device::Type::kCambricon, 1>;
template class Operator<AdamAccumulateGrad, Device::Type::kAscend, 1>;
template class Operator<AdamAccumulateGrad, Device::Type::kMetax, 1>;
template class Operator<AdamAccumulateGrad, Device::Type::kMoore, 1>;
template class Operator<AdamAccumulateGrad, Device::Type::kIluvatar, 1>;
template class Operator<AdamAccumulateGrad, Device::Type::kHygon, 1>;
template class Operator<AdamAccumulateGrad, Device::Type::kMars, 1>;

}  // namespace infini::ops