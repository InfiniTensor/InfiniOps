#ifndef INFINI_OPS_KUNLUN_W4_MARLIN_GEMM_KERNEL_H_
#define INFINI_OPS_KUNLUN_W4_MARLIN_GEMM_KERNEL_H_

#include <stdexcept>

#include "base/w4_marlin_gemm.h"

namespace infini::ops {

template <>
class Operator<W4MarlinGemm, Device::Type::kKunlun> : public W4MarlinGemm {
 public:
  using W4MarlinGemm::W4MarlinGemm;

  [[noreturn]] void operator()(const Tensor a, const Tensor b,
                               const Tensor scales, const Tensor zeros,
                               const Tensor g_idx, const Tensor perm,
                               const Tensor bias, const int64_t scheme,
                               const int64_t group_size, const bool is_k_full,
                               const float alpha, Tensor out) const override {
    throw std::runtime_error("W4MarlinGemm is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
