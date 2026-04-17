#ifndef INFINI_OPS_ILUVATAR_MATMUL_KERNEL_H_
#define INFINI_OPS_ILUVATAR_MATMUL_KERNEL_H_

#include <optional>

#include "base/matmul.h"
#include "iluvatar/gemm/cublas.h"

namespace infini::ops {

template <>
class Operator<Matmul, Device::Type::kIluvatar> : public Matmul {
 public:
  Operator(const Tensor a, const Tensor b, Tensor c, bool trans_a, bool trans_b)
      : Matmul(a, b, c, trans_a, trans_b),
        gemm_(a, b, std::optional<float>{1.0f}, std::optional<float>{0.0f},
              std::optional<int>{static_cast<int>(trans_a)},
              std::optional<int>{static_cast<int>(trans_b)}, c) {}

  void operator()(const Tensor a, const Tensor b, Tensor c, bool trans_a,
                  bool trans_b) const override {
    ConfigureSubOperator(gemm_);
    gemm_(a, b, std::optional<float>{1.0f}, std::optional<float>{0.0f},
          std::optional<int>{static_cast<int>(trans_a)},
          std::optional<int>{static_cast<int>(trans_b)}, c);
  }

 private:
  template <typename Op>
  void ConfigureSubOperator(Op& op) const {
    op.set_handle(handle_);
    op.set_config(config_);
    op.set_stream(stream_);
    op.set_workspace(workspace_);
    op.set_workspace_size_in_bytes(workspace_size_in_bytes_);
  }

  mutable Operator<Gemm, Device::Type::kIluvatar> gemm_;
};

}  // namespace infini::ops

#endif
