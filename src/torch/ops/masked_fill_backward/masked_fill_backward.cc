#include "torch/ops/masked_fill_backward/masked_fill_backward.h"

#include <vector>

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<MaskedFillBackward, kDev, 1>::Operator(const Tensor grad_output,
                                                const Tensor mask,
                                                Tensor grad_input)
    : MaskedFillBackward{grad_output, mask, grad_input} {}

template <Device::Type kDev>
void Operator<MaskedFillBackward, kDev, 1>::operator()(
    const Tensor grad_output, const Tensor mask, Tensor grad_input) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};

  auto at_grad_output = ToAtenTensor<kDev>(
      const_cast<void*>(grad_output.data()), grad_output_shape_,
      grad_output_strides_, grad_output_type_, device_index_);
  auto at_mask = ToAtenTensor<kDev>(const_cast<void*>(mask.data()), mask_shape_,
                                    mask_strides_, mask_type_, device_index_);
  at_mask = at_mask.to(at::kBool);
  auto at_grad_input =
      ToAtenTensor<kDev>(grad_input.data(), grad_input_shape_,
                         grad_input_strides_, grad_input_type_, device_index_);

  auto masked_grad = at::masked_fill(at_grad_output, at_mask, 0);
  const std::vector<int64_t> grad_input_sizes(grad_input_shape_.begin(),
                                              grad_input_shape_.end());
  auto result = masked_grad.sum_to_size(grad_input_sizes);

  at_grad_input.copy_(result);
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(MaskedFillBackward, 1)

}  // namespace infini::ops
