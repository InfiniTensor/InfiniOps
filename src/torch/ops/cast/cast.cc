#include "torch/ops/cast/cast.h"
#include "torch/tensor_.h"
#include "torch/encoder_stream.h"

namespace infini::ops {
template <Device::Type kDev>
void Operator<Cast, kDev, 8>::operator()(const Tensor input, Tensor out) const {
  WithEncoderStream<kDev>(stream_, out.device().index(), [&] {
    auto x = ToAtenTensor<kDev>(const_cast<void*>(input.data()), input.shape(),
        input.strides(), input.dtype(), input.device().index());
    auto y = ToAtenTensor<kDev>(out.data(), out.shape(), out.strides(),
        out.dtype(), out.device().index());
    y.copy_(x);
  });
}
#ifdef WITH_CPU
template class Operator<Cast, Device::Type::kCpu, 8>;
#endif
#ifdef WITH_NVIDIA
template class Operator<Cast, Device::Type::kNvidia, 8>;
#endif
#ifdef WITH_CAMBRICON
template class Operator<Cast, Device::Type::kCambricon, 8>;
#endif
#ifdef WITH_ASCEND
template class Operator<Cast, Device::Type::kAscend, 8>;
#endif
}
