#include "torch/ops/moe_fused_dense/moe_fused_dense.h"
#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"
#include <ATen/Functions.h>
namespace infini::ops {
template <Device::Type kDev>
void Operator<MoeFusedDense, kDev, 2>::operator()(
    Tensor output, const Tensor hidden, const Tensor w13, const Tensor w2,
    const Tensor weights, const Tensor ids, const Tensor sorted,
    const Tensor experts, const Tensor total, size_t block_size,
    const std::optional<Tensor> bias1, const std::optional<Tensor> bias2) const {
  this->Invocation(output, hidden, w13, w2, weights, ids, sorted, experts, total,
                   block_size, bias1, bias2);
  const int device = output.device().index();
  const TorchStreamGuard<kDev> guard{this->stream_, device};
  const auto view = [device](const Tensor &x) {
    return ToAtenTensor<kDev>(const_cast<void *>(x.data()), x.shape(), x.strides(), x.dtype(), device);
  };
  auto x = view(hidden), a = view(w13), b = view(w2), route = view(weights);
  auto y = view(output);
  const auto n = x.size(0), k = ids.size(1), h = x.size(1), mid = b.size(2);
  // The alignment metadata carries local expert IDs after EP remapping.
  // Raw top-k IDs can still be global, so reconstruct the per-pair local IDs.
  auto pair_ids = view(sorted).to(at::kLong);
  auto positions = at::arange(pair_ids.numel(), pair_ids.options());
  auto active = positions.lt(view(total).to(at::kLong).reshape({1}))
                    .logical_and(pair_ids.ge(0)).logical_and(pair_ids.lt(n * k));
  auto active_pairs = pair_ids.masked_select(active);
  auto blocks = at::floor_divide(positions.masked_select(active), static_cast<int64_t>(block_size));
  auto local_experts = view(experts).to(at::kLong).index_select(0, blocks);
  auto indices = at::full({n * k}, -1, pair_ids.options());
  indices.index_copy_(0, active_pairs, local_experts);
  indices = indices.view({n, k});
  // Bound temporary expert gathers to one token's top-k, independent of prefill length.
  for (int64_t token = 0; token < n; ++token) {
    auto expert = indices.select(0, token);
    auto valid = expert.ge(0).logical_and(expert.lt(a.size(0)));
    auto safe_expert = expert.clamp(0, a.size(0) - 1);
    auto wa = a.index_select(0, safe_expert);
    auto wb = b.index_select(0, safe_expert);
    auto input = x.narrow(0, token, 1).unsqueeze(0).expand({k, 1, h}).contiguous();
    auto gate_up = at::bmm(input, wa.transpose(1, 2)).squeeze(1);
    if (bias1) {
      at::Tensor accumulated;
      if constexpr (kDev == Device::Type::kCpu) {
        accumulated = at::bmm(input.to(at::kFloat), wa.transpose(1, 2).to(at::kFloat));
      } else {
        accumulated = at::bmm(input, wa.transpose(1, 2), at::kFloat);
      }
      gate_up = (accumulated.squeeze(1) + view(*bias1).index_select(0, safe_expert).to(at::kFloat)).to(x.scalar_type());
    }
    auto gate = gate_up.narrow(1, 0, mid).to(at::kFloat);
    auto up = gate_up.narrow(1, mid, mid).to(at::kFloat);
    auto activated = (at::silu(gate) * up).to(x.scalar_type()).unsqueeze(1);
    at::Tensor down;
    if constexpr (kDev == Device::Type::kCpu) {
      down = at::bmm(activated.to(at::kFloat), wb.transpose(1, 2).to(at::kFloat));
    } else {
      down = at::bmm(activated, wb.transpose(1, 2), at::kFloat);
    }
    down = down.squeeze(1);
    if (bias2) down = down + view(*bias2).index_select(0, safe_expert).to(at::kFloat);
    auto scaled = down * route.select(0, token).unsqueeze(1);
    auto weighted = at::where(valid.unsqueeze(1), scaled, at::zeros_like(scaled)).to(x.scalar_type());
    y.select(0, token).copy_(weighted.to(at::kFloat).sum(0));
  }
}
INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(MoeFusedDense, 2)
}  // namespace infini::ops
