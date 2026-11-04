#include "torch/ops/transpose/transpose.h"

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<Transpose, kDev, 1>::Operator(const Tensor input, const int64_t dim0,
                                       const int64_t dim1, Tensor out)
    : Transpose{input, dim0, dim1, out}, device_index_{out.device().index()} {}

template <Device::Type kDev>
void Operator<Transpose, kDev, 1>::operator()(const Tensor input,
                                              const int64_t dim0,
                                              const int64_t dim1,
                                              Tensor out) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  auto at_input =
      ToAtenTensor<kDev>(const_cast<void*>(input.data()), input_shape_,
                         input_strides_, input_type_, device_index_);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   out_type_, device_index_);

  at_out.copy_(at_input.transpose(dim0_, dim1_));
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(Transpose, 1)

}  // namespace infini::ops
