#include "torch/random_sample/random_sample.h"

#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<RandomSample, kDev, 1>::Operator(const Tensor logits, float random_val,
                                          float topp, int topk,
                                          float temperature, Tensor out)
    : RandomSample{logits,      random_val, topp, topk,
                   temperature, out},
      device_index_{out.device().index()} {}

template <Device::Type kDev>
void Operator<RandomSample, kDev, 1>::operator()(const Tensor logits,
                                                 float random_val, float topp,
                                                 int topk, float temperature,
                                                 Tensor out) const {
  auto at_logits =
      ToAtenTensor<kDev>(const_cast<void*>(logits.data()), logits_shape_,
                         logits_strides_, logits_type_, device_index_);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   out_type_, device_index_);

  // Greedy path: argmax. `temperature == 0` implies deterministic selection.
  if (topk <= 1 || temperature == 0.0f) {
    auto argmax = at::argmax(at_logits, /*dim=*/0);
    at_out.copy_(argmax);
    return;
  }

  const auto vocab_size = at_logits.size(0);
  const auto effective_topk = std::min<int64_t>(topk, vocab_size);

  // Apply temperature and convert to probabilities in fp32 for stability.
  auto scaled = at_logits.to(at::kFloat) / temperature;
  auto probs = at::softmax(scaled, /*dim=*/0);

  // Top-k truncation.
  auto topk_out = at::topk(probs, effective_topk, /*dim=*/0);
  auto topk_probs = std::get<0>(topk_out);
  auto topk_indices = std::get<1>(topk_out);

  // Top-p truncation. Keep the smallest prefix whose cumulative probability
  // reaches `topp`; always keep at least the top entry.
  auto cumsum = at::cumsum(topk_probs, /*dim=*/0);
  auto keep = cumsum.lt(topp);
  // Shift the mask right so the first entry that crosses `topp` is still
  // kept, matching the "include-one-over" convention.
  auto keep_prefix = at::cat(
      {at::ones({1}, keep.options()), keep.slice(/*dim=*/0, 0, -1)});
  auto filtered = at::where(keep_prefix, topk_probs, at::zeros_like(topk_probs));
  auto total = filtered.sum();
  // Guard against a zero total (can only happen with a pathological input)
  // by falling back to the top-1 entry.
  if (total.template item<float>() == 0.0f) {
    at_out.copy_(topk_indices.index({0}));
    return;
  }
  filtered = filtered / total;

  // Inverse-CDF sample using `random_val`.
  auto cdf = at::cumsum(filtered, /*dim=*/0);
  auto rv = at::scalar_tensor(random_val, cdf.options());
  auto selected = at::searchsorted(cdf, rv);
  // `selected` is a 0D int64 tensor. Clamp into range in case `random_val`
  // equals exactly `1.0f`.
  selected = at::clamp(selected, /*min=*/0, /*max=*/effective_topk - 1);
  at_out.copy_(topk_indices.index({selected}).to(at_out.dtype()));
}

template class Operator<RandomSample, Device::Type::kCpu, 1>;
template class Operator<RandomSample, Device::Type::kNvidia, 1>;
template class Operator<RandomSample, Device::Type::kCambricon, 1>;
template class Operator<RandomSample, Device::Type::kAscend, 1>;
template class Operator<RandomSample, Device::Type::kMetax, 1>;
template class Operator<RandomSample, Device::Type::kMoore, 1>;
template class Operator<RandomSample, Device::Type::kIluvatar, 1>;
template class Operator<RandomSample, Device::Type::kKunlun, 1>;
template class Operator<RandomSample, Device::Type::kHygon, 1>;
template class Operator<RandomSample, Device::Type::kQy, 1>;

}  // namespace infini::ops
