#include "torch/ops/embedding_backward/embedding_backward.h"

#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<EmbeddingBackward, kDev, 1>::Operator(
    const Tensor grad_output, const Tensor indices, const int64_t num_weights,
    const int64_t padding_idx, const bool scale_grad_by_freq, const bool sparse,
    Tensor grad_weight)
    : EmbeddingBackward{grad_output, indices, num_weights, padding_idx,
                        scale_grad_by_freq, sparse, grad_weight},
      device_index_{grad_weight.device().index()} {}

template <Device::Type kDev>
void Operator<EmbeddingBackward, kDev, 1>::operator()(
    const Tensor grad_output, const Tensor indices, const int64_t num_weights,
    const int64_t padding_idx, const bool scale_grad_by_freq, const bool sparse,
    Tensor grad_weight) const {
  auto at_grad_output =
      ToAtenTensor<kDev>(const_cast<void*>(grad_output.data()),
                         grad_output_shape_, grad_output_strides_,
                         grad_output_type_, device_index_);
  auto at_indices = ToAtenTensor<kDev>(const_cast<void*>(indices.data()),
                                       indices_shape_, indices_strides_,
                                       indices_type_, device_index_)
                        .to(at::kLong);  
  auto at_grad_weight =
      ToAtenTensor<kDev>(grad_weight.data(), grad_weight_shape_,
                         grad_weight_strides_, grad_weight_type_, device_index_);

  auto at_grad = at::embedding_dense_backward(at_grad_output, at_indices,
                                              num_weights, padding_idx,
                                              scale_grad_by_freq);
  if (sparse) {
    at_grad = at::embedding_backward(at_grad_output, at_indices, num_weights,
                                     padding_idx, scale_grad_by_freq, true)
                  .to_dense();
  }
  at_grad_weight.copy_(at_grad);
}

template class Operator<EmbeddingBackward, Device::Type::kCpu, 1>;
template class Operator<EmbeddingBackward, Device::Type::kNvidia, 1>;
template class Operator<EmbeddingBackward, Device::Type::kCambricon, 1>;
template class Operator<EmbeddingBackward, Device::Type::kAscend, 1>;
template class Operator<EmbeddingBackward, Device::Type::kMetax, 1>;
template class Operator<EmbeddingBackward, Device::Type::kMoore, 1>;
template class Operator<EmbeddingBackward, Device::Type::kIluvatar, 1>;
template class Operator<EmbeddingBackward, Device::Type::kHygon, 1>;
template class Operator<EmbeddingBackward, Device::Type::kMars, 1>;

}  // namespace infini::ops