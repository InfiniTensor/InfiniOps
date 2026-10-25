#ifdef WITH_TORCH
#include "native/cuda/hygon/ops/top_k_top_p_sampling_from_logits/torch.h"

#include <climits>
#include <random>
#include <stdexcept>
#include <vector>

#include "native/cuda/nvidia/ops/binary_view_capture.h"
#include "torch/hygon/c10.h"
#include "torch/tensor_.h"

namespace infini::ops {
bool Operator<TopKTopPSamplingFromLogits, Device::Type::kHygon, 8>::supports(
    HYGON_SAMPLING_ARGS) {
  if (logits.device().type() != Device::Type::kHygon ||
      out.device() != logits.device() || logits.ndim() != 2 ||
      !IsFloatDtype(logits.dtype()) || !logits.size(0) || !logits.size(1) ||
      logits.size(1) > INT_MAX || out.ndim() != 1 || !out.size(0) ||
      !IsIntegerDtype(out.dtype()) || top_k.ndim() != 1 || top_p.ndim() != 1 ||
      top_k.size(0) != out.size(0) || top_p.size(0) != out.size(0) ||
      !IsIntegerDtype(top_k.dtype()) || !IsFloatDtype(top_p.dtype()) ||
      !out.IsContiguous() || (offset && *offset < 0) ||
      (filter_apply_order != "top_k_first" && filter_apply_order != "joint"))
    return false;
  const auto control = [&](Tensor t) {
    return t.device().type() == Device::Type::kCpu ||
           t.device() == logits.device();
  };
  if (!control(top_k) || !control(top_p)) return false;
  if (indices) {
    if (indices->ndim() != 1 || indices->size(0) != out.size(0) ||
        !IsIntegerDtype(indices->dtype()) || indices->dtype() != out.dtype() ||
        !control(*indices))
      return false;
  } else if (out.size(0) != logits.size(0) || out.dtype() != DataType::kInt32) {
    return false;
  }
  std::size_t output_bytes = 0;
  if (!nvidia_binary_view::SpanBytes(out, &output_bytes)) return false;
  const auto input = [&](Tensor t) {
    std::size_t bytes = 0;
    return nvidia_binary_view::SpanBytes(t, &bytes) &&
           (t.device() != out.device() ||
            !nvidia_binary_view::Overlaps(out, output_bytes, t, bytes));
  };
  return input(logits) && input(top_k) && input(top_p) &&
         (!indices || input(*indices));
}

Operator<TopKTopPSamplingFromLogits, Device::Type::kHygon, 8>::Operator(
    HYGON_SAMPLING_ARGS)
    : TopKTopPSamplingFromLogits(Checked(HYGON_SAMPLING_VALUES), top_k, top_p,
                                 indices, filter_apply_order, deterministic,
                                 check_nan, seed, offset, out) {}

void Operator<TopKTopPSamplingFromLogits, Device::Type::kHygon, 8>::operator()(
    HYGON_SAMPLING_ARGS) const {
  if (!supports(HYGON_SAMPLING_VALUES) || logits.size(1) != vocab_size_ ||
      logits.dtype() != dtype_ || out.size(0) != batch_size_)
    throw std::invalid_argument("unsupported Hygon sampling operands");
  for (const auto& tensor : {logits, top_k, top_p, out})
    if (!tensor.data())
      throw std::invalid_argument("Hygon sampling requires storage");
  if (indices && !indices->data())
    throw std::invalid_argument("Hygon sampling indices require storage");
  const C10<Device::Type::kHygon>::StreamGuard guard{
      C10<Device::Type::kHygon>::GetStreamFromExternal(
          stream_, logits.device().index())};
  const auto view = [](Tensor t) {
    if (t.device().type() == Device::Type::kCpu)
      return ToAtenTensor<Device::Type::kCpu>(
          const_cast<void*>(t.data()), t.shape(), t.strides(), t.dtype(), 0);
    return ToAtenTensor<Device::Type::kHygon>(const_cast<void*>(t.data()),
                                              t.shape(), t.strides(), t.dtype(),
                                              t.device().index());
  };
  auto x = view(logits);
  // Blocking CPU-to-device copies consume borrowed request controls eagerly.
  auto k = view(top_k).to(x.device(), at::kLong).unsqueeze(1);
  auto p = view(top_p).to(x.device(), at::kDouble).unsqueeze(1);
  if (indices) {
    auto rows = view(*indices).to(x.device(), at::kLong);
    if (rows.lt(0).any().item<bool>() || rows.ge(x.size(0)).any().item<bool>())
      throw std::invalid_argument("Hygon sampling row index out of range");
    x = x.index_select(0, rows);
  }
  if (check_nan && x.isnan().any().item<bool>())
    throw std::invalid_argument("Hygon sampling logits contain NaN");
  if (!p.isfinite().all().item<bool>() || p.lt(0).any().item<bool>() ||
      p.gt(1).any().item<bool>())
    throw std::invalid_argument("Hygon sampling top_p must be finite in [0,1]");
  auto sorted = at::sort(x.to(at::kDouble), -1, true);
  auto values = std::get<0>(sorted), vocabulary = std::get<1>(sorted);
  auto weights = (values - values.select(1, 0).unsqueeze(1)).exp();
  auto rank = at::arange(vocab_size_, k.options()).unsqueeze(0);
  k = at::where(k.le(0) | k.gt(int64_t(vocab_size_)), int64_t(vocab_size_), k);
  auto topk_mask = rank.lt(k);
  auto normalizer_weights =
      filter_apply_order == "top_k_first" ? weights * topk_mask : weights;
  auto cdf_before = normalizer_weights.cumsum(1) - normalizer_weights;
  auto cutoff = p * normalizer_weights.sum(1, true);
  auto keep = topk_mask & (cdf_before.lt(cutoff) | rank.eq(0));
  weights = weights * keep;
  auto cdf = weights.cumsum(1) / weights.sum(1, true);
  const uint64_t actual_seed =
      seed ? uint64_t(*seed) : uint64_t(std::random_device{}());
  std::mt19937_64 rng(actual_seed);
  rng.discard(uint64_t(offset.value_or(0)));
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::vector<double> draws(batch_size_);
  for (auto& draw : draws) draw = uniform(rng);
  auto random = at::tensor(draws, at::TensorOptions().dtype(at::kDouble))
                    .to(x.device())
                    .unsqueeze(1);
  auto position =
      cdf.lt(random).sum(1, true).clamp_max(int64_t(vocab_size_ - 1));
  view(out).copy_(vocabulary.gather(1, position).squeeze(1));
  (void)deterministic;  // Seed and offset fully specify this provider's RNG.
}
}  // namespace infini::ops
#endif
