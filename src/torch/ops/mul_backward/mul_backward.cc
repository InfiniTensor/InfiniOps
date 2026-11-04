#include "torch/ops/mul_backward/mul_backward.h"

#include <cassert>
#include <cstdint>
#include <vector>

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {
template <Device::Type kDev>
Operator<MulBackward, kDev, 1>::Operator(const Tensor grad_output,
                                         const Tensor a, const Tensor b,
                                         Tensor grad_a, Tensor grad_b)
    : MulBackward{grad_output, a, b, grad_a, grad_b},
      device_index_{grad_a.device().index()} {}

template <Device::Type kDev>
void Operator<MulBackward, kDev, 1>::operator()(const Tensor grad_output,
                                                const Tensor a, const Tensor b,
                                                Tensor grad_a,
                                                Tensor grad_b) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  auto at_grad = ToAtenTensor<kDev>(const_cast<void*>(grad_output.data()),
                                    grad_output_shape_, grad_output_strides_,
                                    grad_output_type_, device_index_);
  auto at_a = ToAtenTensor<kDev>(const_cast<void*>(a.data()), a_shape_,
                                 a_strides_, a_type_, device_index_);
  auto at_b = ToAtenTensor<kDev>(const_cast<void*>(b.data()), b_shape_,
                                 b_strides_, b_type_, device_index_);
  auto at_grad_a =
      ToAtenTensor<kDev>(grad_a.data(), grad_a_shape_, grad_a_strides_,
                         grad_a_type_, device_index_);
  auto at_grad_b =
      ToAtenTensor<kDev>(grad_b.data(), grad_b_shape_, grad_b_strides_,
                         grad_b_type_, device_index_);

  at_grad_a.copy_((at_grad * at_b)
                      .sum_to_size(std::vector<int64_t>(grad_a_shape_.begin(),
                                                        grad_a_shape_.end())));
  at_grad_b.copy_((at_grad * at_a)
                      .sum_to_size(std::vector<int64_t>(grad_b_shape_.begin(),
                                                        grad_b_shape_.end())));
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(MulBackward, 1)

}  // namespace infini::ops
