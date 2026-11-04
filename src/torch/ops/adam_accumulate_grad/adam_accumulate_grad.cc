#include "torch/ops/adam_accumulate_grad/adam_accumulate_grad.h"

#include <cmath>

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<AdamAccumulateGrad, kDev, 1>::Operator(
    const Tensor grad, const double learning_rate, const double beta1,
    const double beta2, const double eps, const int64_t step, Tensor param,
    Tensor m, Tensor v)
    : AdamAccumulateGrad{grad, learning_rate, beta1, beta2, eps,
                         step, param,         m,     v},
      device_index_{param.device().index()} {}

template <Device::Type kDev>
Operator<AdamAccumulateGrad, kDev, 1>::Operator(
    const Tensor grad, Tensor param, Tensor m, Tensor v,
    const double learning_rate, const double beta1, const double beta2,
    const double eps, const int64_t step)
    : Operator{grad, learning_rate, beta1, beta2, eps, step, param, m, v} {}

template <Device::Type kDev>
void Operator<AdamAccumulateGrad, kDev, 1>::operator()(
    const Tensor grad, const double learning_rate, const double beta1,
    const double beta2, const double eps, const int64_t step, Tensor param,
    Tensor m, Tensor v) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  auto at_grad = ToAtenTensor<kDev>(const_cast<void*>(grad.data()), grad_shape_,
                                    grad_strides_, grad_type_, device_index_);
  auto at_param = ToAtenTensor<kDev>(param.data(), param_shape_, param_strides_,
                                     param_type_, device_index_);
  auto at_m = ToAtenTensor<kDev>(m.data(), m_shape_, m_strides_, m_type_,
                                 device_index_);
  auto at_v = ToAtenTensor<kDev>(v.data(), v_shape_, v_strides_, v_type_,
                                 device_index_);

  const auto compute_type =
      at_param.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
  const auto gradient = at_grad.to(compute_type);
  const auto next_m =
      at_m.to(compute_type) * beta1_ + gradient * (1.0 - beta1_);
  const auto next_v =
      at_v.to(compute_type) * beta2_ + gradient.square() * (1.0 - beta2_);
  const double bcm = 1.0 - std::pow(beta1_, static_cast<double>(step_));
  const double bcv = 1.0 - std::pow(beta2_, static_cast<double>(step_));
  const auto update =
      learning_rate_ * (next_m / bcm) / (at::sqrt(next_v / bcv) + eps_);
  const auto next_param = at_param.to(compute_type) - update;
  at_param.copy_(next_param);
  at_m.copy_(next_m);
  at_v.copy_(next_v);
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(AdamAccumulateGrad, 1)

}  // namespace infini::ops
