#include "torch/ops/pad/pad.h"

#include "torch/c10.h"
#ifdef WITH_NVIDIA
#include "torch/nvidia/c10.h"
#endif
#ifdef WITH_CAMBRICON
#include "torch/cambricon/c10.h"
#endif
#ifdef WITH_ASCEND
#include "torch/ascend/c10.h"
#endif
#ifdef WITH_METAX
#include "torch/metax/c10.h"
#endif
#ifdef WITH_MOORE
#include "torch/moore/c10.h"
#endif
#ifdef WITH_ILUVATAR
#include "torch/iluvatar/c10.h"
#endif
#ifdef WITH_HYGON
#include "torch/hygon/c10.h"
#endif
#ifdef WITH_MARS
#include "torch/mars/c10.h"
#endif
#ifdef WITH_THEAD
#include "torch/thead/c10.h"
#endif
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
void Operator<Pad, kDev, 1>::operator()(const Tensor input,
                                        const std::vector<int64_t> pad,
                                        const std::string mode,
                                        const std::optional<double> value,
                                        Tensor out) const {
  const auto run = [&] {
    auto at_input = ToAtenTensor<kDev>(
        const_cast<void*>(input.data()), input_meta_.shape(),
        input_meta_.strides(), input_meta_.dtype(), device_.index());
    auto at_out =
        ToAtenTensor<kDev>(out.data(), out_meta_.shape(), out_meta_.strides(),
                           out_meta_.dtype(), device_.index());
    // The functional overload also supports older vendor PyTorch releases.
    // Its temporary stays on the device; the caller's output is never resized.
    at_out.copy_(at::constant_pad_nd(at_input, pad, value.value_or(0.0)));
  };
  if constexpr (kDev != Device::Type::kCpu) {
    const typename C10<kDev>::StreamGuard stream_guard{
        C10<kDev>::GetStreamFromExternal(stream_, device_.index())};
    run();
  } else {
    run();
  }
}

#ifdef WITH_CPU
template class Operator<Pad, Device::Type::kCpu, 1>;
#endif
#ifdef WITH_NVIDIA
template class Operator<Pad, Device::Type::kNvidia, 1>;
#endif
#ifdef WITH_CAMBRICON
template class Operator<Pad, Device::Type::kCambricon, 1>;
#endif
#ifdef WITH_ASCEND
template class Operator<Pad, Device::Type::kAscend, 1>;
#endif
#ifdef WITH_METAX
template class Operator<Pad, Device::Type::kMetax, 1>;
#endif
#ifdef WITH_MOORE
template class Operator<Pad, Device::Type::kMoore, 1>;
#endif
#ifdef WITH_ILUVATAR
template class Operator<Pad, Device::Type::kIluvatar, 1>;
#endif
#ifdef WITH_HYGON
template class Operator<Pad, Device::Type::kHygon, 1>;
#endif
#ifdef WITH_MARS
template class Operator<Pad, Device::Type::kMars, 1>;
#endif
#ifdef WITH_THEAD
template class Operator<Pad, Device::Type::kThead, 1>;
#endif

}  // namespace infini::ops
