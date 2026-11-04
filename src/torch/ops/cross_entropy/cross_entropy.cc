#include "torch/ops/cross_entropy/cross_entropy.h"

#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<CrossEntropy, kDev, 1>::Operator(const Tensor input,
                                          const Tensor target, Tensor out)
    : CrossEntropy{input, target, out}, device_index_{out.device().index()} {}

template <Device::Type kDev>
void Operator<CrossEntropy, kDev, 1>::operator()(const Tensor input,
                                                 const Tensor target,
                                                 Tensor out) const {
  auto at_input = ToAtenTensor<kDev>(const_cast<void*>(input.data()),
                                     input_shape_, input_strides_, input_type_,
                                     device_index_);
  auto at_target = ToAtenTensor<kDev>(const_cast<void*>(target.data()),
                                      target_shape_, target_strides_,
                                      target_type_, device_index_)
                       .to(at::kLong);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   out_type_, device_index_);

  // 默认参数：reduction=Mean, ignore_index=-100, label_smoothing=0 —— 与 InfiniTrain 一致
  auto loss = at::cross_entropy_loss(at_input, at_target);
  at_out.copy_(loss);
}

template class Operator<CrossEntropy, Device::Type::kCpu, 1>;
template class Operator<CrossEntropy, Device::Type::kNvidia, 1>;
template class Operator<CrossEntropy, Device::Type::kCambricon, 1>;
template class Operator<CrossEntropy, Device::Type::kAscend, 1>;
template class Operator<CrossEntropy, Device::Type::kMetax, 1>;
template class Operator<CrossEntropy, Device::Type::kMoore, 1>;
template class Operator<CrossEntropy, Device::Type::kIluvatar, 1>;
template class Operator<CrossEntropy, Device::Type::kHygon, 1>;
template class Operator<CrossEntropy, Device::Type::kMars, 1>;

}  // namespace infini::ops