#include "torch/ops/repeat_interleave/repeat_interleave.h"

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<RepeatInterleave, kDev, 1>::Operator(const Tensor input,
                                              const int64_t repeats,
                                              const int64_t dim, Tensor out)
    : RepeatInterleave{input, repeats, dim, out},
      device_index_{out.device().index()} {}

template <Device::Type kDev>
void Operator<RepeatInterleave, kDev, 1>::operator()(const Tensor input,
                                                     const int64_t repeats,
                                                     const int64_t dim,
                                                     Tensor out) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  auto at_input =
      ToAtenTensor<kDev>(const_cast<void*>(input.data()), input_shape_,
                         input_strides_, input_type_, device_index_);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   out_type_, device_index_);

  at_out.copy_(at::repeat_interleave(at_input, repeats_, dim_));
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(RepeatInterleave, 1)

}  // namespace infini::ops
