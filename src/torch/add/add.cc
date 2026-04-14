#include "torch/add/add.h"

#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<Add, kDev, 1>::Operator(const Tensor input, const Tensor other,
                                 Tensor out)
    : Add{input, other, out},
      device_index_{out.device().index()} {}

template <Device::Type kDev>
void Operator<Add, kDev, 1>::operator()(const Tensor input, const Tensor other,
                                        Tensor out) const {
  auto at_input = ToAtenTensor<kDev>(
      const_cast<void*>(input.data()), input_shape_, input_strides_,
      input_type_, device_index_);
  auto at_other = ToAtenTensor<kDev>(
      const_cast<void*>(other.data()), other_shape_, other_strides_,
      other_type_, device_index_);
  auto at_out = ToAtenTensor<kDev>(
      out.data(), out_shape_, out_strides_, out_type_, device_index_);

  at::add_out(at_out, at_input, at_other);
}

template class Operator<Add, Device::Type::kCpu, 1>;
template class Operator<Add, Device::Type::kNvidia, 1>;
template class Operator<Add, Device::Type::kCambricon, 1>;
template class Operator<Add, Device::Type::kAscend, 1>;
template class Operator<Add, Device::Type::kMetax, 1>;
template class Operator<Add, Device::Type::kMoore, 1>;
template class Operator<Add, Device::Type::kIluvatar, 1>;
template class Operator<Add, Device::Type::kKunlun, 1>;
template class Operator<Add, Device::Type::kHygon, 1>;
template class Operator<Add, Device::Type::kQy, 1>;

}  // namespace infini::ops
