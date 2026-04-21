#include "torch/swiglu/swiglu.h"

#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<Swiglu, kDev, 1>::Operator(const Tensor input, const Tensor gate,
                                    Tensor out)
    : Swiglu{input, gate, out}, device_index_{out.device().index()} {}

template <Device::Type kDev>
void Operator<Swiglu, kDev, 1>::operator()(const Tensor input,
                                           const Tensor gate,
                                           Tensor out) const {
  auto at_input =
      ToAtenTensor<kDev>(const_cast<void*>(input.data()), input_shape_,
                         input_strides_, input_type_, device_index_);
  auto at_gate =
      ToAtenTensor<kDev>(const_cast<void*>(gate.data()), gate_shape_,
                         gate_strides_, gate_type_, device_index_);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   out_type_, device_index_);

  // `SwiGLU(input, gate) = input * SiLU(gate)`.
  at::mul_out(at_out, at_input, at::silu(at_gate));
}

template class Operator<Swiglu, Device::Type::kCpu, 1>;
template class Operator<Swiglu, Device::Type::kNvidia, 1>;
template class Operator<Swiglu, Device::Type::kCambricon, 1>;
template class Operator<Swiglu, Device::Type::kAscend, 1>;
template class Operator<Swiglu, Device::Type::kMetax, 1>;
template class Operator<Swiglu, Device::Type::kMoore, 1>;
template class Operator<Swiglu, Device::Type::kIluvatar, 1>;
template class Operator<Swiglu, Device::Type::kKunlun, 1>;
template class Operator<Swiglu, Device::Type::kHygon, 1>;
template class Operator<Swiglu, Device::Type::kQy, 1>;

}  // namespace infini::ops
