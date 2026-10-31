#include "torch/ops/add/add.h"

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
  const auto run = [&] {
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
  };
  if constexpr (kDev != Device::Type::kCpu) {
    const typename C10<kDev>::StreamGuard guard{
        C10<kDev>::GetStreamFromExternal(stream_, device_index_)};
    run();
  } else {
    run();
  }
}

#ifdef WITH_CPU
template class Operator<Add, Device::Type::kCpu, 1>;
#endif
#ifdef WITH_NVIDIA
template class Operator<Add, Device::Type::kNvidia, 1>;
#endif
#ifdef WITH_CAMBRICON
template class Operator<Add, Device::Type::kCambricon, 1>;
#endif
#ifdef WITH_ASCEND
template class Operator<Add, Device::Type::kAscend, 1>;
#endif
#ifdef WITH_METAX
template class Operator<Add, Device::Type::kMetax, 1>;
#endif
#ifdef WITH_MOORE
template class Operator<Add, Device::Type::kMoore, 1>;
#endif
#ifdef WITH_ILUVATAR
template class Operator<Add, Device::Type::kIluvatar, 1>;
#endif
#ifdef WITH_HYGON
template class Operator<Add, Device::Type::kHygon, 1>;
#endif
#ifdef WITH_MARS
template class Operator<Add, Device::Type::kMars, 1>;
#endif

}  // namespace infini::ops
