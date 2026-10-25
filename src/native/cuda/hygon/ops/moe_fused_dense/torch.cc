// Adapted from InfiniCore's Hygon ATen MoE fallback; preserve device stream.
#ifdef WITH_TORCH
#include "native/cuda/hygon/ops/moe_fused_dense/torch.h"

#include <map>
#include <vector>

#include "torch/hygon/c10.h"
#include "torch/tensor_.h"
namespace infini::ops {
void Operator<MoeFusedDense, Device::Type::kHygon, 8>::operator()(
    Tensor output, const Tensor hidden, const Tensor w13, const Tensor w2,
    const Tensor weights, const Tensor ids, const Tensor sorted,
    const Tensor experts, const Tensor total, size_t block_size,
    const std::optional<Tensor> bias1,
    const std::optional<Tensor> bias2) const {
  Invocation(output, hidden, w13, w2, weights, ids, sorted, experts, total,
             block_size, bias1, bias2);
  const C10<Device::Type::kHygon>::StreamGuard guard{
      C10<Device::Type::kHygon>::GetStreamFromExternal(
          stream_, output.device().index())};
  const auto view = [](Tensor t) {
    return ToAtenTensor<Device::Type::kHygon>(const_cast<void*>(t.data()),
                                              t.shape(), t.strides(), t.dtype(),
                                              t.device().index());
  };
  auto out = view(output), x = view(hidden), up_weights = view(w13),
       down_weights = view(w2);
  auto routing = view(weights), expert_ids = view(ids);
  auto result = at::zeros(out.sizes(), out.options().dtype(at::kFloat));
  const auto intermediate = w2.size(2);
  // One small route-table transfer replaces E*K nonzero synchronizations.
  // Group all routes for an active expert into one BLAS batch.
  const auto host_ids = expert_ids.reshape({-1}).to(at::kCPU);
  const auto* id_data = host_ids.data_ptr<int32_t>();
  std::map<int64_t, std::vector<int64_t>> routes;
  for (int64_t pair = 0; pair < host_ids.numel(); ++pair) {
    const auto expert = id_data[pair];
    if (expert < 0 || expert >= up_weights.size(0))
      throw std::invalid_argument("Hygon MoE route expert is out of range");
    routes[expert].push_back(pair);
  }
  for (const auto& [expert, pairs] : routes) {
    auto route =
        at::tensor(pairs, at::TensorOptions().dtype(at::kLong)).to(x.device());
    auto tokens = at::floor_divide(route, expert_ids.size(1));
    auto selected = x.index_select(0, tokens);
    auto up_matrix = up_weights.select(0, expert).transpose(0, 1);
    auto gate_up =
        bias1 ? at::addmm(view(*bias1).select(0, expert), selected, up_matrix)
              : at::matmul(selected, up_matrix);
    // Match native SiLU/multiply: float arithmetic, one final activation cast.
    auto activated =
        (at::silu(gate_up.narrow(1, 0, intermediate).to(at::kFloat)) *
         gate_up.narrow(1, intermediate, intermediate).to(at::kFloat))
            .to(x.scalar_type());
    auto down_matrix = down_weights.select(0, expert).transpose(0, 1);
    auto values = bias2 ? at::addmm(view(*bias2).select(0, expert), activated,
                                    down_matrix)
                        : at::matmul(activated, down_matrix);
    auto scale = routing.reshape({-1}).index_select(0, route).unsqueeze(1);
    // Native MoeFusedDense weights and sums routes in float, then casts once.
    result.index_add_(0, tokens, values.to(at::kFloat) * scale);
  }
  out.copy_(result);
}
}  // namespace infini::ops
#endif
