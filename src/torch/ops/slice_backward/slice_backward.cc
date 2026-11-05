#include "torch/ops/slice_backward/slice_backward.h"

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<SliceBackward, kDev, 1>::Operator(const Tensor grad_output,
                                           const std::vector<int64_t> starts,
                                           const std::vector<int64_t> ends,
                                           const std::vector<int64_t> steps,
                                           Tensor grad_input)
    : SliceBackward{grad_output, starts, ends, steps, grad_input},
      device_index_{grad_input.device().index()} {}

template <Device::Type kDev>
void Operator<SliceBackward, kDev, 1>::operator()(
    const Tensor grad_output, const std::vector<int64_t> starts,
    const std::vector<int64_t> ends, const std::vector<int64_t> steps,
    Tensor grad_input) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  using namespace torch::indexing;

  auto at_grad_output = ToAtenTensor<kDev>(
      const_cast<void*>(grad_output.data()), grad_output_shape_,
      grad_output_strides_, grad_output_type_, device_index_);
  auto at_grad_input =
      ToAtenTensor<kDev>(grad_input.data(), grad_input_shape_,
                         grad_input_strides_, grad_input_type_, device_index_);

  std::vector<TensorIndex> index;
  index.reserve(starts_.size());
  for (std::size_t i = 0; i < starts_.size(); ++i) {
    index.emplace_back(at::indexing::Slice(starts_[i], ends_[i], steps_[i]));
  }

  auto zeros = at::zeros(at_grad_input.sizes(), at_grad_input.options());
  zeros.index_put_(index, at_grad_output);
  at_grad_input.copy_(zeros);
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(SliceBackward, 1)

}  // namespace infini::ops
