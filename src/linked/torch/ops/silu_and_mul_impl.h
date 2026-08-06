#ifndef INFINI_OPS_LINKED_TORCH_OPS_SILU_AND_MUL_IMPL_H_
#define INFINI_OPS_LINKED_TORCH_OPS_SILU_AND_MUL_IMPL_H_

#include "linked/torch/ops/silu_and_mul.h"
#include "torch/tensor_.h"

namespace infini::ops::linked::torch {

template <typename Backend>
TorchSiluAndMul<Backend>::TorchSiluAndMul(const Tensor input, Tensor out)
    : ::infini::ops::SiluAndMul{input, out},
      device_index_{out.device().index()} {}

template <typename Backend>
void TorchSiluAndMul<Backend>::operator()(const Tensor input, Tensor out) const {
  const typename Backend::StreamGuard stream_guard{
      Backend::GetStreamFromExternal(stream_, device_index_)};
  auto at_input = ToAtenTensor<Backend::kDeviceType>(
      const_cast<void*>(input.data()), input_shape_, input_strides_, input_type_,
      device_index_);
  auto at_out = ToAtenTensor<Backend::kDeviceType>(
      out.data(), out_shape_, out_strides_, out_type_, device_index_);

  auto provider_input = is_input_contiguous_ ? at_input : at_input.contiguous();
  auto provider_out =
      is_out_contiguous_ ? at_out : at::empty(at_out.sizes(), at_out.options());

  Backend::Call(provider_out, provider_input);
  if (!is_out_contiguous_) {
    at_out.copy_(provider_out);
  }
}

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_OPS_SILU_AND_MUL_IMPL_H_
