#ifndef INFINI_OPS_NVIDIA_PREPARE_MOE_INPUT_KERNEL_H_
#define INFINI_OPS_NVIDIA_PREPARE_MOE_INPUT_KERNEL_H_

#include <optional>

#include "base/prepare_moe_input.h"

namespace infini::ops {

template <>
class Operator<PrepareMoeInput, Device::Type::kNvidia, 0>
    : public PrepareMoeInput {
 public:
  using PrepareMoeInput::PrepareMoeInput;

  using PrepareMoeInput::operator();

  void operator()(const Tensor topk_ids, const int64_t num_experts,
                  const int64_t n, const int64_t k, Tensor expert_offsets,
                  Tensor problem_sizes1, Tensor problem_sizes2,
                  Tensor input_permutation, Tensor output_permutation,
                  std::optional<Tensor> blockscale_offsets) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_PREPARE_MOE_INPUT_KERNEL_H_
