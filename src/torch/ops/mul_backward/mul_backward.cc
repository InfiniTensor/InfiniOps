#include "torch/ops/mul_backward/mul_backward.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "torch/instantiate.h"
#include "torch/storage_overlap.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<MulBackward, kDev, 1>::Operator(const Tensor grad_output,
                                         const Tensor input, const Tensor other,
                                         Tensor grad_input, Tensor grad_other)
    : MulBackward{grad_output, input, other, grad_input, grad_other} {}

template <Device::Type kDev>
void Operator<MulBackward, kDev, 1>::operator()(const Tensor grad_output,
                                                const Tensor input,
                                                const Tensor other,
                                                Tensor grad_input,
                                                Tensor grad_other) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};

  auto at_grad = ToAtenTensor<kDev>(const_cast<void*>(grad_output.data()),
                                    grad_output_shape_, grad_output_strides_,
                                    grad_output_type_, device_index_);
  auto at_input =
      ToAtenTensor<kDev>(const_cast<void*>(input.data()), input_shape_,
                         input_strides_, input_type_, device_index_);
  auto at_other =
      ToAtenTensor<kDev>(const_cast<void*>(other.data()), other_shape_,
                         other_strides_, other_type_, device_index_);
  auto at_grad_input =
      ToAtenTensor<kDev>(grad_input.data(), grad_input_shape_,
                         grad_input_strides_, grad_input_type_, device_index_);
  auto at_grad_other =
      ToAtenTensor<kDev>(grad_other.data(), grad_other_shape_,
                         grad_other_strides_, grad_other_type_, device_index_);

  assert(!StorageSpansOverlap(at_grad_input, at_grad_other) &&
         "`MulBackward` output storage must not overlap.");

  const std::vector<int64_t> grad_input_sizes(grad_input_shape_.begin(),
                                              grad_input_shape_.end());
  const std::vector<int64_t> grad_other_sizes(grad_other_shape_.begin(),
                                              grad_other_shape_.end());

  auto grad_input_result = (at_grad * at_other).sum_to_size(grad_input_sizes);
  auto grad_other_result = (at_grad * at_input).sum_to_size(grad_other_sizes);

  at_grad_input.copy_(grad_input_result);
  at_grad_other.copy_(grad_other_result);
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(MulBackward, 1)

}  // namespace infini::ops
