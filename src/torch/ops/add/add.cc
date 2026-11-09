#include "torch/ops/add/add.h"

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<Add, kDev, 1>::Operator(const Tensor input, const Tensor other,
                                 const double alpha, Tensor out)
    : Add{input, other, alpha, out}, device_index_{out.device().index()} {}

template <Device::Type kDev>
Operator<Add, kDev, 1>::Operator(const Tensor input, const Tensor other,
                                 Tensor out)
    : Operator{input, other, 1.0, out} {}

template <Device::Type kDev>
void Operator<Add, kDev, 1>::operator()(const Tensor input, const Tensor other,
                                        const double alpha, Tensor out) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  auto at_input =
      ToAtenTensor<kDev>(const_cast<void*>(input.data()), input_shape_,
                         input_strides_, input_type_, device_index_);
  auto at_other =
      ToAtenTensor<kDev>(const_cast<void*>(other.data()), other_shape_,
                         other_strides_, other_type_, device_index_);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   out_type_, device_index_);

  auto at_alpha = at_input.is_floating_point()
                      ? at::Scalar(alpha)
                      : at::Scalar(static_cast<int64_t>(alpha));
  at::add_out(at_out, at_input, at_other, at_alpha);
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(Add, 1)

}  // namespace infini::ops
