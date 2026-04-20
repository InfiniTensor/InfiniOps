#include "torch/embedding/embedding.h"

#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<Embedding, kDev, 1>::Operator(const Tensor input, const Tensor weight,
                                       Tensor out)
    : Embedding{input, weight, out}, device_index_{out.device().index()} {}

template <Device::Type kDev>
void Operator<Embedding, kDev, 1>::operator()(const Tensor input,
                                              const Tensor weight,
                                              Tensor out) const {
  auto at_input =
      ToAtenTensor<kDev>(const_cast<void*>(input.data()), input_shape_,
                         input_strides_, input_type_, device_index_);
  auto at_weight =
      ToAtenTensor<kDev>(const_cast<void*>(weight.data()), weight_shape_,
                         weight_strides_, weight_type_, device_index_);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   out_type_, device_index_);

  // `ATen` provides no `embedding_out`, so we compute via `index_select_out`
  // on a flattened view of `out`, which writes directly into `out`'s storage.
  auto at_input_flat = at_input.reshape(-1);
  auto at_out_flat = at_out.view({-1, at_weight.size(-1)});
  at::index_select_out(at_out_flat, at_weight, 0, at_input_flat);
}

template class Operator<Embedding, Device::Type::kCpu, 1>;
template class Operator<Embedding, Device::Type::kNvidia, 1>;
template class Operator<Embedding, Device::Type::kCambricon, 1>;
template class Operator<Embedding, Device::Type::kAscend, 1>;
template class Operator<Embedding, Device::Type::kMetax, 1>;
template class Operator<Embedding, Device::Type::kMoore, 1>;
template class Operator<Embedding, Device::Type::kIluvatar, 1>;
template class Operator<Embedding, Device::Type::kKunlun, 1>;
template class Operator<Embedding, Device::Type::kHygon, 1>;
template class Operator<Embedding, Device::Type::kQy, 1>;

}  // namespace infini::ops
