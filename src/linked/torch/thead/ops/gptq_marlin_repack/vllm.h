#ifndef INFINI_OPS_LINKED_TORCH_THEAD_OPS_GPTQ_MARLIN_REPACK_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_THEAD_OPS_GPTQ_MARLIN_REPACK_VLLM_H_

#include "linked/torch/ops/gptq_marlin_repack.h"
#include "torch/thead/c10.h"

namespace infini::ops::linked::torch::thead {

struct VllmGptqMarlinRepack : C10<Device::Type::kThead> {
  static at::Tensor Call(at::Tensor b_q_weight, at::Tensor perm, int64_t size_k,
                         int64_t size_n, int64_t num_bits, bool is_a_8bit);
};

}  // namespace infini::ops::linked::torch::thead

namespace infini::ops::linked::torch {

extern template class TorchGptqMarlinRepack<
    ::infini::ops::linked::torch::thead::VllmGptqMarlinRepack>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<GptqMarlinRepack, Device::Type::kThead, 16>
    : public linked::torch::TorchGptqMarlinRepack<
          linked::torch::thead::VllmGptqMarlinRepack> {
 public:
  using linked::torch::TorchGptqMarlinRepack<
      linked::torch::thead::VllmGptqMarlinRepack>::TorchGptqMarlinRepack;

  using linked::torch::TorchGptqMarlinRepack<
      linked::torch::thead::VllmGptqMarlinRepack>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_THEAD_OPS_GPTQ_MARLIN_REPACK_VLLM_H_
