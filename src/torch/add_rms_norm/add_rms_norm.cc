#include "torch/add_rms_norm/add_rms_norm.h"

#include "torch/tensor_.h"

#include <ATen/ops/add.h>
#include <ATen/ops/rms_norm.h>

#include <vector>

namespace infini::ops {

template <Device::Type kDev>
Operator<AddRmsNorm, kDev, 1>::Operator(const Tensor input, const Tensor other,
                                        const Tensor weight, float eps,
                                        Tensor out, Tensor residual_out)
    : AddRmsNorm{input, other, weight, eps, out, residual_out},
      device_index_{out.device().index()} {}

template <Device::Type kDev>
void Operator<AddRmsNorm, kDev, 1>::operator()(
    const Tensor input, const Tensor other, const Tensor weight, float eps,
    Tensor out, Tensor residual_out) const {
  auto at_input =
      ToAtenTensor<kDev>(const_cast<void*>(input.data()), input_shape_,
                         input_strides_, input_type_, device_index_);
  auto at_other =
      ToAtenTensor<kDev>(const_cast<void*>(other.data()), input_shape_,
                         other_strides_, other_type_, device_index_);
  auto at_weight =
      ToAtenTensor<kDev>(const_cast<void*>(weight.data()), weight_shape_,
                         weight_strides_, weight_type_, device_index_);
  auto at_out = ToAtenTensor<kDev>(out.data(), input_shape_, out_strides_,
                                   out_type_, device_index_);
  auto at_residual_out =
      ToAtenTensor<kDev>(residual_out.data(), input_shape_, residual_out_strides_,
                         residual_out_type_, device_index_);

  // `residual_out = input + other`. Writing directly into the caller-owned
  // buffer eliminates the temporary allocation/copy that composing `Add` +
  // `RmsNorm` would incur.
  at::add_out(at_residual_out, at_input, at_other);

  // `out = rms_norm(residual_out, weight, eps)`. ATen has no `rms_norm_out`,
  // so we copy the result into the caller's buffer.
  const std::vector<int64_t> normalized_shape(weight_shape_.begin(),
                                              weight_shape_.end());
  at_out.copy_(at::rms_norm(at_residual_out, normalized_shape, at_weight,
                            static_cast<double>(eps)));
}

template class Operator<AddRmsNorm, Device::Type::kCpu, 1>;
template class Operator<AddRmsNorm, Device::Type::kNvidia, 1>;
template class Operator<AddRmsNorm, Device::Type::kCambricon, 1>;
template class Operator<AddRmsNorm, Device::Type::kAscend, 1>;
template class Operator<AddRmsNorm, Device::Type::kMetax, 1>;
template class Operator<AddRmsNorm, Device::Type::kMoore, 1>;
template class Operator<AddRmsNorm, Device::Type::kIluvatar, 1>;
template class Operator<AddRmsNorm, Device::Type::kKunlun, 1>;
template class Operator<AddRmsNorm, Device::Type::kHygon, 1>;
template class Operator<AddRmsNorm, Device::Type::kQy, 1>;

}  // namespace infini::ops
