#include "torch/ops/mean_backward/mean_backward.h"

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<MeanBackward, kDev, 1>::Operator(const Tensor grad_output,
                                          const int64_t dim,
                                          const bool keep_dim,
                                          Tensor grad_input)
    : MeanBackward{grad_output, dim, keep_dim, grad_input},
      device_index_{grad_input.device().index()} {}

template <Device::Type kDev>
void Operator<MeanBackward, kDev, 1>::operator()(const Tensor grad_output,
                                                 const int64_t dim,
                                                 const bool keep_dim,
                                                 Tensor grad_input) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  auto at_grad_output = ToAtenTensor<kDev>(
      const_cast<void*>(grad_output.data()), grad_output_shape_,
      grad_output_strides_, grad_output_type_, device_index_);
  auto at_grad_input =
      ToAtenTensor<kDev>(grad_input.data(), grad_input_shape_,
                         grad_input_strides_, grad_input_type_, device_index_);

  if (at_grad_input.numel() == 0) {
    return;
  }

  auto at_g = at_grad_output;
  if (!keep_dim_) {
    at_g = at_g.unsqueeze(dim_);
  }

  const double count = static_cast<double>(grad_input_shape_[dim_]);
  at_grad_input.copy_(at_g.expand(at_grad_input.sizes()) / count);
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(MeanBackward, 1)

}  // namespace infini::ops
