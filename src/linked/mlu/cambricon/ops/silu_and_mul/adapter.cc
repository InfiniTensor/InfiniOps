#include "linked/mlu/cambricon/ops/silu_and_mul/adapter.h"

#include "linked/mlu/torch_context.h"
#include "torch/tensor_.h"

// Cambricon Apex `glu_activation` exports this global C++ symbol.
at::Tensor bias_swiglu_fwd(const at::Tensor& input, const at::Tensor& bias);

namespace infini::ops {

Operator<SiluAndMul, Device::Type::kCambricon, 11>::Operator(const Tensor input,
                                                             Tensor out)
    : SiluAndMul{input, out}, device_index_{out.device().index()} {}

void Operator<SiluAndMul, Device::Type::kCambricon, 11>::operator()(
    const Tensor input, Tensor out) const {
  const linked::mlu::TorchContextGuard context_guard{stream_, device_index_};
  auto at_input = ToAtenTensor<Device::Type::kCambricon>(
      const_cast<void*>(input.data()), input_shape_, input_strides_,
      input_type_, device_index_);
  auto at_out = ToAtenTensor<Device::Type::kCambricon>(
      out.data(), out_shape_, out_strides_, out_type_, device_index_);

  auto provider_input = is_input_contiguous_ ? at_input : at_input.contiguous();
  const auto bias = at::empty({0}, at::TensorOptions().dtype(at::kInt));
  const auto result = ::bias_swiglu_fwd(provider_input, bias);
  at_out.copy_(result);
}

}  // namespace infini::ops
