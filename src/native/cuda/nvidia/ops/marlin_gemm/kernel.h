// SPDX-License-Identifier: Apache-2.0

#ifndef INFINI_OPS_NVIDIA_MARLIN_GEMM_KERNEL_H_
#define INFINI_OPS_NVIDIA_MARLIN_GEMM_KERNEL_H_

#include <optional>

#include "base/marlin_gemm.h"

namespace infini::ops {

template <>
class Operator<MarlinGemm, Device::Type::kNvidia, 0> : public MarlinGemm {
 public:
  Operator(const Tensor a, const Tensor b_q_weight,
           std::optional<Tensor> b_bias, const Tensor b_scales,
           std::optional<Tensor> a_scales, std::optional<Tensor> global_scale,
           std::optional<Tensor> b_zeros, std::optional<Tensor> g_idx,
           std::optional<Tensor> perm, const Tensor workspace,
           int64_t b_type_id, int64_t size_m, int64_t size_n, int64_t size_k,
           bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
           bool is_zp_float, Tensor out);

  void operator()(const Tensor a, const Tensor b_q_weight,
                  std::optional<Tensor> b_bias, const Tensor b_scales,
                  std::optional<Tensor> a_scales,
                  std::optional<Tensor> global_scale,
                  std::optional<Tensor> b_zeros, std::optional<Tensor> g_idx,
                  std::optional<Tensor> perm, const Tensor workspace,
                  int64_t b_type_id, int64_t size_m, int64_t size_n,
                  int64_t size_k, bool is_k_full, bool use_atomic_add,
                  bool use_fp32_reduce, bool is_zp_float,
                  Tensor out) const override;

 private:
  int sms_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_MARLIN_GEMM_KERNEL_H_
