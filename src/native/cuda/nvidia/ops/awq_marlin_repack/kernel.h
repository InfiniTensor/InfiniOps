#ifndef INFINI_OPS_NVIDIA_AWQ_MARLIN_REPACK_KERNEL_H_
#define INFINI_OPS_NVIDIA_AWQ_MARLIN_REPACK_KERNEL_H_

#include "base/awq_marlin_repack.h"

namespace infini::ops {

template <>
class Operator<AwqMarlinRepack, Device::Type::kNvidia, 0>
    : public AwqMarlinRepack {
 public:
  using AwqMarlinRepack::AwqMarlinRepack;

  void operator()(const Tensor b_q_weight, const int64_t size_k,
                  const int64_t size_n, const int64_t num_bits,
                  const bool is_a_8bit, Tensor out) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_AWQ_MARLIN_REPACK_KERNEL_H_
