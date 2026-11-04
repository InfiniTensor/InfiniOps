#include "torch/ops/transpose_backward/transpose_backward.h"

#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<TransposeBackward, kDev, 1>::Operator(
    const Tensor grad_output, const int64_t dim0, const int64_t dim1,
    Tensor grad_input)
    : TransposeBackward{grad_output, dim0, dim1, grad_input},
      device_index_{grad_input.device().index()} {}

template <Device::Type kDev>
void Operator<TransposeBackward, kDev, 1>::operator()(
    const Tensor grad_output, const int64_t dim0, const int64_t dim1,
    Tensor grad_input) const {
  auto at_grad_output = ToAtenTensor<kDev>(
      const_cast<void*>(grad_output.data()), grad_output_shape_,
      grad_output_strides_, grad_output_type_, device_index_);

  auto at_grad_input = ToAtenTensor<kDev>(
      grad_input.data(), grad_input_shape_, grad_input_strides_,
      grad_input_type_, device_index_);

  at_grad_input.copy_(at_grad_output.transpose(dim0_, dim1_));
}

template class Operator<TransposeBackward, Device::Type::kCpu, 1>;
template class Operator<TransposeBackward, Device::Type::kNvidia, 1>;
template class Operator<TransposeBackward, Device::Type::kCambricon, 1>;
template class Operator<TransposeBackward, Device::Type::kAscend, 1>;
template class Operator<TransposeBackward, Device::Type::kMetax, 1>;
template class Operator<TransposeBackward, Device::Type::kMoore, 1>;
template class Operator<TransposeBackward, Device::Type::kIluvatar, 1>;
template class Operator<TransposeBackward, Device::Type::kHygon, 1>;
template class Operator<TransposeBackward, Device::Type::kMars, 1>;

}  // namespace infini::ops