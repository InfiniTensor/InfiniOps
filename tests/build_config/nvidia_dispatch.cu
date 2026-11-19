#include "native/cuda/nvidia/ops/quick_gelu/kernel.h"

namespace infini::ops {

bool ProbeQuickGeluSelection(const Config& config, Tensor input,
                             Tensor output) {
  return Operator<QuickGelu>::CanMake(config, input, output);
}

std::unique_ptr<Operator<QuickGelu>> ProbeQuickGeluConstruction(
    const Config& config, Tensor input, Tensor output) {
  return Operator<QuickGelu>::Make(config, input, output);
}

}  // namespace infini::ops
