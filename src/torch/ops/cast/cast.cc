#include "torch/ops/cast/cast.h"

#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {
template <Device::Type kDev>
void Operator<Cast, kDev, 8>::operator()(const Tensor input, Tensor out) const {
  TORCH_CHECK(supports(input, out), "cast: unsupported metadata or alias");
  if (torch_detail::SameView(input, out)) return;
  WithTorchStream<kDev>(stream_, out.device().index(), [&] {
    auto x = ToAtenTensor<kDev>(const_cast<void*>(input.data()), input.shape(),
                                input.strides(), input.dtype(),
                                input.device().index());
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
#ifdef WITH_METAX
template class Operator<Cast, Device::Type::kMetax, 8>;
#endif
#ifdef WITH_MOORE
template class Operator<Cast, Device::Type::kMoore, 8>;
#endif
#ifdef WITH_ILUVATAR
template class Operator<Cast, Device::Type::kIluvatar, 8>;
#endif
#ifdef WITH_HYGON
template class Operator<Cast, Device::Type::kHygon, 8>;
#endif
#ifdef WITH_THEAD
template class Operator<Cast, Device::Type::kThead, 8>;
#endif
#ifdef WITH_MARS
template class Operator<Cast, Device::Type::kMars, 8>;
#endif
}  // namespace infini::ops
