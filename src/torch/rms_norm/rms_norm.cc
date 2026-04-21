#include "torch/rms_norm/rms_norm.h"

#include "torch/tensor_.h"

#include <ATen/ops/rms_norm.h>

#include <vector>

namespace infini::ops {

template <Device::Type kDev>
Operator<RmsNorm, kDev, 1>::Operator(const Tensor input, const Tensor weight,
                                     float eps, Tensor out)
    : RmsNorm{input, weight, eps, out},
      device_index_{out.device().index()} {}

template <Device::Type kDev>
void Operator<RmsNorm, kDev, 1>::operator()(const Tensor input,
                                            const Tensor weight, float eps,
                                            Tensor out) const {
  auto at_input =
      ToAtenTensor<kDev>(const_cast<void*>(input.data()), input_shape_,
                         input_strides_, input.dtype(), device_index_);
  auto at_weight =
      ToAtenTensor<kDev>(const_cast<void*>(weight.data()), weight.shape(),
                         weight.strides(), weight.dtype(), device_index_);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   out.dtype(), device_index_);

  const std::vector<int64_t> normalized_shape(weight.shape().begin(),
                                              weight.shape().end());
  at_out.copy_(at::rms_norm(at_input, normalized_shape, at_weight,
                            static_cast<double>(eps)));
}

template class Operator<RmsNorm, Device::Type::kCpu, 1>;
template class Operator<RmsNorm, Device::Type::kNvidia, 1>;
template class Operator<RmsNorm, Device::Type::kCambricon, 1>;
template class Operator<RmsNorm, Device::Type::kAscend, 1>;
template class Operator<RmsNorm, Device::Type::kMetax, 1>;
template class Operator<RmsNorm, Device::Type::kMoore, 1>;
template class Operator<RmsNorm, Device::Type::kIluvatar, 1>;
template class Operator<RmsNorm, Device::Type::kKunlun, 1>;
template class Operator<RmsNorm, Device::Type::kHygon, 1>;
template class Operator<RmsNorm, Device::Type::kQy, 1>;

}  // namespace infini::ops
