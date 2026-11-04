#include "torch/ops/add_backward/add_backward.h"

#include <cassert>
#include <cstdint>
#include <vector>

#include "torch/tensor_.h"

namespace infini::ops {
namespace {

at::Tensor ReduceToShape(const at::Tensor& grad,
                         const std::vector<int64_t>& target_shape) {
  const int64_t ndim = static_cast<int64_t>(target_shape.size());
  const int64_t grad_ndim = grad.dim();
  assert(grad_ndim >= ndim && "grad rank must not be smaller than target rank");

  std::vector<int64_t> sum_dims;
  for (int64_t i = 0; i < grad_ndim; ++i) {
    const int64_t target_i = i - (grad_ndim - ndim);  
    if (target_i < 0) {                               
      sum_dims.push_back(i);
      continue;
    }
    if (target_shape[target_i] == 1 && grad.size(i) != 1) {
      sum_dims.push_back(i);
    }
  }

  if (sum_dims.empty()) {
    return grad.reshape(target_shape);
  }
  return grad.sum(sum_dims, /*keepdim=*/true).reshape(target_shape);
}

}  // namespace

template <Device::Type kDev>
Operator<AddBackward, kDev, 1>::Operator(const Tensor grad_output,
                                         Tensor grad_a, Tensor grad_b)
    : AddBackward{grad_output, grad_a, grad_b},
      device_index_{grad_a.device().index()} {}

template <Device::Type kDev>
void Operator<AddBackward, kDev, 1>::operator()(const Tensor grad_output,
                                                Tensor grad_a,
                                                Tensor grad_b) const {
  auto at_grad = ToAtenTensor<kDev>(const_cast<void*>(grad_output.data()),
                                    grad_output_shape_, grad_output_strides_,
                                    grad_output_type_, device_index_);
  auto at_grad_a = ToAtenTensor<kDev>(grad_a.data(), grad_a_shape_,
                                      grad_a_strides_, grad_a_type_,
                                      device_index_);
  auto at_grad_b = ToAtenTensor<kDev>(grad_b.data(), grad_b_shape_,
                                      grad_b_strides_, grad_b_type_,
                                      device_index_);

  at_grad_a.copy_(ReduceToShape(
      at_grad, std::vector<int64_t>(grad_a_shape_.begin(), grad_a_shape_.end())));
  at_grad_b.copy_(ReduceToShape(
      at_grad, std::vector<int64_t>(grad_b_shape_.begin(), grad_b_shape_.end())));
}

template class Operator<AddBackward, Device::Type::kCpu, 1>;
template class Operator<AddBackward, Device::Type::kNvidia, 1>;
template class Operator<AddBackward, Device::Type::kCambricon, 1>;
template class Operator<AddBackward, Device::Type::kAscend, 1>;
template class Operator<AddBackward, Device::Type::kMetax, 1>;
template class Operator<AddBackward, Device::Type::kMoore, 1>;
template class Operator<AddBackward, Device::Type::kIluvatar, 1>;
template class Operator<AddBackward, Device::Type::kHygon, 1>;
template class Operator<AddBackward, Device::Type::kMars, 1>;

}  // namespace infini::ops