#include "linked/cuda/metax/ops/silu_and_mul/adapter.h"

#include "linked/cuda/torch_context.h"
#include "torch/tensor_.h"

// MetaX vLLM `_C` exports this global C++ symbol.
void silu_and_mul(at::Tensor& out, at::Tensor& input);

namespace infini::ops {

Operator<SiluAndMul, Device::Type::kMetax, 11>::Operator(const Tensor input,
                                                         Tensor out)
    : SiluAndMul{input, out}, device_index_{out.device().index()} {}

void Operator<SiluAndMul, Device::Type::kMetax, 11>::operator()(
    const Tensor input, Tensor out) const {
  const linked::cuda::TorchContextGuard context_guard{stream_, device_index_};
  auto at_input = ToAtenTensor<Device::Type::kMetax>(
      const_cast<void*>(input.data()), input_shape_, input_strides_, input_type_,
      device_index_);
  auto at_out = ToAtenTensor<Device::Type::kMetax>(
      out.data(), out_shape_, out_strides_, out_type_, device_index_);

  auto provider_input =
      is_input_contiguous_ ? at_input : at_input.contiguous();
  auto provider_out =
      is_out_contiguous_ ? at_out : at::empty(at_out.sizes(), at_out.options());

  ::silu_and_mul(provider_out, provider_input);
  if (!is_out_contiguous_) {
    at_out.copy_(provider_out);
  }
}

}  // namespace infini::ops
