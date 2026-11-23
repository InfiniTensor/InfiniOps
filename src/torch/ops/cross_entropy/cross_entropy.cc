#include "torch/ops/cross_entropy/cross_entropy.h"

#include <ATen/core/Reduction.h>

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<CrossEntropy, kDev, 1>::Operator(const Tensor input,
                                          const Tensor target,
                                          std::optional<Tensor> weight,
                                          int64_t ignore_index,
                                          std::string reduction,
                                          double label_smoothing, Tensor out)
    : CrossEntropy{input,     target,          weight, ignore_index,
                   reduction, label_smoothing, out} {}

template <Device::Type kDev>
void Operator<CrossEntropy, kDev, 1>::operator()(
    const Tensor input, const Tensor target, std::optional<Tensor> weight,
    int64_t ignore_index, std::string reduction, double label_smoothing,
    Tensor out) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  auto at_input =
      ToAtenTensor<kDev>(const_cast<void*>(input.data()), input_shape_,
                         input_strides_, input_type_, device_index_);
  auto at_target =
      ToAtenTensor<kDev>(const_cast<void*>(target.data()), target_shape_,
                         target_strides_, target_type_, device_index_);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   out_type_, device_index_);

  std::optional<at::Tensor> at_weight;
  if (weight) {
    at_weight =
        ToAtenTensor<kDev>(const_cast<void*>(weight->data()), weight->shape(),
                           weight->strides(), weight->dtype(), device_index_);
  }

  const bool probability_target = target_shape_ == input_shape_;
  if (!probability_target) at_target = at_target.to(at::kLong);

  const auto promote_low_precision = [](at::Tensor value) {
    if (value.scalar_type() == at::kHalf ||
        value.scalar_type() == at::kBFloat16) {
      return value.to(at::kFloat);
    }
    return value;
  };
  at_input = promote_low_precision(at_input);
  if (probability_target) at_target = promote_low_precision(at_target);
  if (at_weight) *at_weight = promote_low_precision(*at_weight);

  assert((reduction == "none" || reduction == "mean" || reduction == "sum") &&
         "Invalid CrossEntropy reduction");
  const int64_t at_reduction = reduction == "none"  ? at::Reduction::None
                               : reduction == "sum" ? at::Reduction::Sum
                                                    : at::Reduction::Mean;

  auto loss =
      at::cross_entropy_loss(at_input, at_target, at_weight, at_reduction,
                             ignore_index, label_smoothing);
  at_out.copy_(loss);
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(CrossEntropy, 1)

}  // namespace infini::ops
