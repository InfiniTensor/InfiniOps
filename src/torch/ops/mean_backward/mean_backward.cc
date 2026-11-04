#include "torch/ops/mean_backward/mean_backward.h"

#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<MeanBackward, kDev, 1>::Operator(const Tensor grad_output,
                                          const int64_t dim, const bool keep_dim,
                                          Tensor grad_input)
    : MeanBackward{grad_output, dim, keep_dim, grad_input},
      device_index_{grad_input.device().index()} {}

template <Device::Type kDev>
void Operator<MeanBackward, kDev, 1>::operator()(const Tensor grad_output,
                                                 const int64_t dim,
                                                 const bool keep_dim,
                                                 Tensor grad_input) const {
  auto at_grad_output =
      ToAtenTensor<kDev>(const_cast<void*>(grad_output.data()),
                         grad_output_shape_, grad_output_strides_,
                         grad_output_type_, device_index_);
  auto at_grad_input =
      ToAtenTensor<kDev>(grad_input.data(), grad_input_shape_,
                         grad_input_strides_, grad_input_type_, device_index_);

  if (grad_input_shape_.empty() || grad_input_shape_[0] == 0) {
    return;  
  }

  auto at_g = at_grad_output;
  if (!keep_dim_) {
    at_g = at_g.unsqueeze(dim_);  
  }

  const double count = static_cast<double>(at_grad_input.numel()) /
                       static_cast<double>(at_grad_output.numel());
  at_grad_input.copy_(at_g.expand(at_grad_input.sizes()) / count);
}

template class Operator<MeanBackward, Device::Type::kCpu, 1>;
template class Operator<MeanBackward, Device::Type::kNvidia, 1>;
template class Operator<MeanBackward, Device::Type::kCambricon, 1>;
template class Operator<MeanBackward, Device::Type::kAscend, 1>;
template class Operator<MeanBackward, Device::Type::kMetax, 1>;
template class Operator<MeanBackward, Device::Type::kMoore, 1>;
template class Operator<MeanBackward, Device::Type::kIluvatar, 1>;
template class Operator<MeanBackward, Device::Type::kHygon, 1>;
template class Operator<MeanBackward, Device::Type::kMars, 1>;

}  // namespace infini::ops