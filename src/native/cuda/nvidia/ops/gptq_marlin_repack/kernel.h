#ifndef INFINI_OPS_NVIDIA_GPTQ_MARLIN_REPACK_KERNEL_H_
#define INFINI_OPS_NVIDIA_GPTQ_MARLIN_REPACK_KERNEL_H_

#include "base/gptq_marlin_repack.h"

namespace infini::ops {

template <>
class Operator<GptqMarlinRepack, Device::Type::kNvidia, 0>
    : public GptqMarlinRepack {
 public:
  using GptqMarlinRepack::GptqMarlinRepack;

  void operator()(const Tensor b_q_weight, const Tensor perm,
                  const int64_t size_k, const int64_t size_n,
                  const int64_t num_bits, const bool is_a_8bit,
                  Tensor out) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_GPTQ_MARLIN_REPACK_KERNEL_H_
