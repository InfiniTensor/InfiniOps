#include "torch/ops/max_pool2d/max_pool2d.h"

#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {
template <Device::Type kDev>
void Operator<MaxPool2d, kDev, 8>::operator()(
    const Tensor input, const std::vector<int64_t> kernel_size,
    const std::vector<int64_t> stride, const std::vector<int64_t> padding,
    const std::vector<int64_t> dilation, bool ceil_mode, Tensor out) const {
  TORCH_CHECK(
      input.device() == out.device() && input.device().type() == kDev &&
          input.dtype() == out.dtype(),
      "MaxPool2d input and output must have matching device and dtype.");
  const TorchStreamGuard<kDev> guard{stream_, out.device().index()};
  auto x = ToAtenTensor<kDev>(const_cast<void*>(input.data()),
                              input_layout_.shape(), input_layout_.strides(),
                              input_layout_.dtype(), input.device().index());
  auto y =
      ToAtenTensor<kDev>(out.data(), out_layout_.shape(), out_layout_.strides(),
                         out_layout_.dtype(), out.device().index());
  const auto result =
      at::max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode);
  TORCH_CHECK(y.sizes() == result.sizes(),
              "MaxPool2d output shape must match the pooling result.");
  y.copy_(result);
}
#ifdef WITH_CPU
template class Operator<MaxPool2d, Device::Type::kCpu, 8>;
#endif
#ifdef WITH_THEAD
template class Operator<MaxPool2d, Device::Type::kThead, 8>;
#endif
#ifdef WITH_NVIDIA
template class Operator<MaxPool2d, Device::Type::kNvidia, 8>;
#endif
#ifdef WITH_CAMBRICON
template class Operator<MaxPool2d, Device::Type::kCambricon, 8>;
#endif
#ifdef WITH_ASCEND
template class Operator<MaxPool2d, Device::Type::kAscend, 8>;
#endif
#ifdef WITH_METAX
template class Operator<MaxPool2d, Device::Type::kMetax, 8>;
#endif
#ifdef WITH_MOORE
template class Operator<MaxPool2d, Device::Type::kMoore, 8>;
#endif
#ifdef WITH_ILUVATAR
template class Operator<MaxPool2d, Device::Type::kIluvatar, 8>;
#endif
#ifdef WITH_HYGON
template class Operator<MaxPool2d, Device::Type::kHygon, 8>;
#endif
#ifdef WITH_MARS
template class Operator<MaxPool2d, Device::Type::kMars, 8>;
#endif
}  // namespace infini::ops
