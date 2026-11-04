#include "torch/ops/vocab_parallel_cross_entropy_backward/vocab_parallel_cross_entropy_backward.h"

#include <cassert>

#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<VocabParallelCrossEntropyBackward, kDev, 1>::Operator(
    const Tensor grad_output, const Tensor softmax_local,
    const Tensor target_mask, const Tensor masked_target,
    const Tensor valid_mask_local, const int64_t vocab_size_local,
    const int64_t vocab_size_original, const double label_smoothing,
    Tensor grad_input)
    : VocabParallelCrossEntropyBackward{
          grad_output, softmax_local, target_mask, masked_target,
          valid_mask_local, vocab_size_local, vocab_size_original,
          label_smoothing, grad_input} {}

template <Device::Type kDev>
void Operator<VocabParallelCrossEntropyBackward, kDev, 1>::operator()(
    const Tensor grad_output, const Tensor softmax_local,
    const Tensor target_mask, const Tensor masked_target,
    const Tensor valid_mask_local, const int64_t vocab_size_local,
    const int64_t vocab_size_original, const double label_smoothing,
    Tensor grad_input) const {
  auto at_grad_output = ToAtenTensor<kDev>(
      const_cast<void*>(grad_output.data()), grad_output_shape_,
      grad_output_strides_, grad_output_type_, device_index_);
  auto at_softmax = ToAtenTensor<kDev>(
      const_cast<void*>(softmax_local.data()), softmax_shape_, softmax_strides_,
      softmax_type_, device_index_);
  auto at_target_mask = ToAtenTensor<kDev>(
      const_cast<void*>(target_mask.data()), target_mask_shape_,
      target_mask_strides_, target_mask_type_, device_index_);
  auto at_masked_target = ToAtenTensor<kDev>(
      const_cast<void*>(masked_target.data()), masked_target_shape_,
      masked_target_strides_, masked_target_type_, device_index_);
  auto at_valid_mask = ToAtenTensor<kDev>(
      const_cast<void*>(valid_mask_local.data()), valid_mask_shape_,
      valid_mask_strides_, valid_mask_type_, device_index_);
  auto at_grad_input = ToAtenTensor<kDev>(
      grad_input.data(), grad_input_shape_, grad_input_strides_,
      grad_input_type_, device_index_);

  assert(vocab_size_local > 0 &&
         "VocabParallelCrossEntropyBackward requires positive local vocab");
  assert(vocab_size_original > 0 &&
         "VocabParallelCrossEntropyBackward requires positive global vocab");
  assert(at_softmax.numel() % vocab_size_local == 0 &&
         "softmax_local size is not divisible by vocab_size_local");
  assert(at_target_mask.numel() == at_softmax.numel() / vocab_size_local &&
         "target_mask size mismatch");
  assert(at_masked_target.numel() == at_target_mask.numel() &&
         "masked_target size mismatch");
  assert(at_valid_mask.numel() == vocab_size_local &&
         "valid_mask_local size mismatch");

  const auto rows = at_softmax.numel() / vocab_size_local;
  assert((at_grad_output.numel() == 1 || at_grad_output.numel() == rows) &&
         "grad_output must be scalar or one value per row");

  auto softmax_2d = at_softmax.reshape({rows, vocab_size_local})
                        .to(at_grad_input.scalar_type());
  auto target_mask_1d =
      at_target_mask.reshape({rows}).to(at_grad_input.scalar_type());
  auto target_1d = at_masked_target.reshape({rows}).to(at::kLong);
  auto valid_1d =
      at_valid_mask.reshape({vocab_size_local}).to(at_grad_input.scalar_type());

  auto dloss = at_grad_output.reshape({-1}).to(at_grad_input.scalar_type());
  if (dloss.numel() == 1) {
    dloss = dloss.expand({rows});
  }

  auto gradient = softmax_2d * dloss.reshape({rows, 1});
  const auto valid_rows = 1.0 - target_mask_1d;
  const auto row_scale = dloss * (1.0 - label_smoothing) * valid_rows;

  auto target_one_hot = at::zeros_like(gradient);
  target_one_hot.scatter_(1, target_1d.reshape({rows, 1}), 1.0);
  gradient.sub_(target_one_hot * row_scale.reshape({rows, 1}));

  const auto smoothing_term =
      label_smoothing / static_cast<double>(vocab_size_original);
  gradient.sub_(dloss.reshape({rows, 1}) * smoothing_term *
                valid_1d.reshape({1, vocab_size_local}));
  gradient.mul_(valid_1d.reshape({1, vocab_size_local}));

  at_grad_input.copy_(gradient.reshape(at_grad_input.sizes()));
}

template class Operator<VocabParallelCrossEntropyBackward,
                        Device::Type::kCpu, 1>;
template class Operator<VocabParallelCrossEntropyBackward,
                        Device::Type::kNvidia, 1>;
template class Operator<VocabParallelCrossEntropyBackward,
                        Device::Type::kCambricon, 1>;
template class Operator<VocabParallelCrossEntropyBackward,
                        Device::Type::kAscend, 1>;
template class Operator<VocabParallelCrossEntropyBackward,
                        Device::Type::kMetax, 1>;
template class Operator<VocabParallelCrossEntropyBackward,
                        Device::Type::kMoore, 1>;
template class Operator<VocabParallelCrossEntropyBackward,
                        Device::Type::kIluvatar, 1>;
template class Operator<VocabParallelCrossEntropyBackward,
                        Device::Type::kHygon, 1>;
template class Operator<VocabParallelCrossEntropyBackward,
                        Device::Type::kMars, 1>;

}  // namespace infini::ops
