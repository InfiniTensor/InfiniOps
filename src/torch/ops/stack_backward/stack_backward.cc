#include "torch/ops/stack_backward/stack_backward.h"

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<StackBackward, kDev, 1>::Operator(const Tensor grad_output,
                                           const int64_t dim,
                                           std::vector<Tensor> outs)
    : StackBackward{grad_output, dim, std::move(outs)},
      device_index_{StackBackward::device_index_} {}

template <Device::Type kDev>
void Operator<StackBackward, kDev, 1>::operator()(
    const Tensor grad_output, const int64_t dim,
    std::vector<Tensor> outs) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  auto at_grad_output = ToAtenTensor<kDev>(
      const_cast<void*>(grad_output.data()), grad_output_shape_,
      grad_output_strides_, grad_output_type_, device_index_);

  const auto parts = at::unbind(at_grad_output, dim_);
  assert(parts.size() == outs_.size() && "`StackBackward` part count mismatch");

  for (std::size_t i = 0; i < parts.size(); ++i) {
    auto at_out_i =
        ToAtenTensor<kDev>(outs[i].data(), out_shapes_[i], out_strides_[i],
                           out_types_[i], device_index_);
    at_out_i.copy_(parts[i]);
  }
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(StackBackward, 1)

}  // namespace infini::ops
