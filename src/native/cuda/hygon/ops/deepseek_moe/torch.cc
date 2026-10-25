#ifdef WITH_TORCH
#include "native/cuda/hygon/ops/deepseek_moe/torch.h"

#include <map>
#include <vector>

#include "torch/hygon/c10.h"
#include "torch/tensor_.h"
namespace infini::ops {
void Operator<DeepseekMoe, Device::Type::kHygon, 8>::operator()(
    Tensor output, const Tensor hidden, const Tensor ids, const Tensor weights,
    const std::vector<Tensor> gate, const std::vector<Tensor> up,
    const std::vector<Tensor> down, size_t intermediate,
    size_t num_experts) const {
  Invocation(output, hidden, ids, weights, gate, up, down, intermediate,
             num_experts);
  const C10<Device::Type::kHygon>::StreamGuard guard{
      C10<Device::Type::kHygon>::GetStreamFromExternal(
          stream_, output.device().index())};
  const auto view = [](Tensor t) {
    return ToAtenTensor<Device::Type::kHygon>(const_cast<void*>(t.data()),
                                              t.shape(), t.strides(), t.dtype(),
                                              t.device().index());
  };
  auto out = view(output), x = view(hidden), routing = view(weights);
  const auto expert_ids = view(ids);
  auto result = at::zeros(out.sizes(), out.options().dtype(at::kFloat));
  // Copy the small route table once, rather than synchronizing each expert.
  const auto host_ids = expert_ids.reshape({-1}).to(at::kCPU);
  const auto* id_data = host_ids.data_ptr<int32_t>();
  std::map<int64_t, std::vector<int64_t>> routes;
  for (int64_t pair = 0; pair < host_ids.numel(); ++pair) {
    const auto expert = id_data[pair];
    if (expert < 0 || size_t(expert) >= num_experts)
      throw std::invalid_argument("Hygon MoE route expert is out of range");
    routes[expert].push_back(pair);
  }
  for (const auto& [expert, pairs] : routes) {
    auto route =
        at::tensor(pairs, at::TensorOptions().dtype(at::kLong)).to(x.device());
    auto tokens = at::floor_divide(route, expert_ids.size(1));
    const auto selected = x.index_select(0, tokens);
    auto g = at::matmul(selected, view(gate[expert]).transpose(0, 1));
    auto u = at::matmul(selected, view(up[expert]).transpose(0, 1));
    auto scale = routing.reshape({-1}).index_select(0, route).unsqueeze(1);
    // Deepseek applies routing weights before the down projection.
    auto activated = (at::silu(g.to(at::kFloat)) * u.to(at::kFloat) * scale)
                         .to(x.scalar_type());
    auto values = at::matmul(activated, view(down[expert]).transpose(0, 1));
    result.index_add_(0, tokens, values.to(at::kFloat));
  }
  // Delay the write to preserve the supported output == hidden alias.
  out.copy_(result);
}
}  // namespace infini::ops
#endif
