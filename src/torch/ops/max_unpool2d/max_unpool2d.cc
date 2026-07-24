#include "torch/ops/max_unpool2d/max_unpool2d.h"

#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
void Operator<MaxUnpool2d, kDev, 8>::Run(const Tensor input,
                                         const Tensor indices,
                                         const std::vector<int64_t> output_size,
                                         Tensor out) const {
  const auto device_index = out.device().index();
  auto at_input =
      ToAtenTensor<kDev>(const_cast<void*>(input.data()), input_shape_,
                         input_strides_, input_type_, device_index);
  auto at_indices =
      ToAtenTensor<kDev>(const_cast<void*>(indices.data()), indices_shape_,
                         indices_strides_, indices_type_, device_index);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   out_type_, device_index);

  at::max_unpool2d_out(at_out, at_input, at_indices, output_size);
}

#ifdef WITH_CPU
template class Operator<MaxUnpool2d, Device::Type::kCpu, 8>;
#endif
#ifdef WITH_NVIDIA
template class Operator<MaxUnpool2d, Device::Type::kNvidia, 8>;
#endif
#ifdef WITH_CAMBRICON
template class Operator<MaxUnpool2d, Device::Type::kCambricon, 8>;
#endif
#ifdef WITH_ASCEND
template class Operator<MaxUnpool2d, Device::Type::kAscend, 8>;
#endif
#ifdef WITH_METAX
template class Operator<MaxUnpool2d, Device::Type::kMetax, 8>;
#endif
#ifdef WITH_MOORE
template class Operator<MaxUnpool2d, Device::Type::kMoore, 8>;
#endif
#ifdef WITH_ILUVATAR
template class Operator<MaxUnpool2d, Device::Type::kIluvatar, 8>;
#endif
#ifdef WITH_HYGON
template class Operator<MaxUnpool2d, Device::Type::kHygon, 8>;
#endif

}  // namespace infini::ops
