#ifndef INFINI_OPS_CPU_MATMUL_MATMUL_H_
#define INFINI_OPS_CPU_MATMUL_MATMUL_H_

#include <optional>

#include "base/matmul.h"
#include "cpu/gemm/gemm.h"

namespace infini::ops {

template <>
class Operator<Matmul, Device::Type::kCpu> : public Matmul {
 public:
  Operator(const Tensor a, const Tensor b, Tensor c, bool trans_a, bool trans_b)
      : Matmul(a, b, c, trans_a, trans_b),
        gemm_(a, b, std::optional<float>{1.0f}, std::optional<float>{0.0f},
              std::optional<int>{static_cast<int>(trans_a)},
              std::optional<int>{static_cast<int>(trans_b)}, c) {}

  void operator()(const Tensor a, const Tensor b, Tensor c, bool trans_a,
                  bool trans_b) const override {
    gemm_(a, b, std::optional<float>{1.0f}, std::optional<float>{0.0f},
          std::optional<int>{static_cast<int>(trans_a)},
          std::optional<int>{static_cast<int>(trans_b)}, c);
  }

 private:
  mutable Operator<Gemm, Device::Type::kCpu> gemm_;
};

}  // namespace infini::ops

#endif
