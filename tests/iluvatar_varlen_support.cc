#include <infini/ops.h>

#include <optional>
#include <vector>

int main() {
  using namespace infini::ops;
  const Device device{Device::Type::kIluvatar, 0};
  const Tensor q{nullptr, {8, 4, 128}, DataType::kBFloat16, device};
  const Tensor flat{nullptr, {8, 2, 128}, DataType::kBFloat16, device};
  const Tensor paged{nullptr, {2, 256, 2, 128}, DataType::kBFloat16, device};
  const Tensor offsets{nullptr, {3}, DataType::kInt32, device};
  const Tensor blocks{nullptr, {2, 1}, DataType::kInt32, device};
  const std::optional<Tensor> none;
  Config config;
  config.set_implementation_index(0);
  const auto supported = [&](Tensor cache, std::optional<Tensor> table,
                             bool causal, std::vector<int64_t> window) {
    return FlashAttnVarlenFunc::CanMake(
        config, q, cache, cache, offsets, offsets, none, table, int64_t{4},
        int64_t{4}, double{0.0}, std::optional<double>{}, causal, window,
        double{0.0}, false, false, q, none, none);
  };
  if (supported(flat, none, true, {-1, -1})) return 1;
  if (!supported(paged, blocks, true, {-1, -1})) return 2;
  if (supported(paged, blocks, false, {-1, -1})) return 3;
  if (supported(paged, blocks, true, {64, 0})) return 4;
  return 0;
}
