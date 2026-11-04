#include "torch/ops/repeat_interleave_backward/repeat_interleave_backward.h"

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<RepeatInterleaveBackward, kDev, 1>::Operator(const Tensor grad_output,
                                                      const int64_t dim,
                                                      Tensor grad_input)
    : RepeatInterleaveBackward{grad_output, dim, grad_input},
      device_index_{grad_input.device().index()} {}

template <Device::Type kDev>
void Operator<RepeatInterleaveBackward, kDev, 1>::operator()(
    const Tensor grad_output, const int64_t dim, Tensor grad_input) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  auto at_grad_output = ToAtenTensor<kDev>(
      const_cast<void*>(grad_output.data()), grad_output_shape_,
      grad_output_strides_, grad_output_type_, device_index_);
  auto at_grad_input =
      ToAtenTensor<kDev>(grad_input.data(), grad_input_shape_,
                         grad_input_strides_, grad_input_type_, device_index_);

  if (at_grad_input.numel() == 0) return;
  if (repeats_ == 0) {
    at_grad_input.zero_();
    return;
  }

  auto sizes = at_grad_output.sizes().vec();
  sizes[dim_] /= repeats_;
  sizes.insert(sizes.begin() + dim_ + 1, repeats_);

  auto grouped = at_grad_output.reshape(sizes);
  auto summed = grouped.sum(dim_ + 1);

  at_grad_input.copy_(summed);
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(RepeatInterleaveBackward, 1)

}  // namespace infini::ops
