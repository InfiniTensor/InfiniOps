#ifndef INFINI_OPS_MOORE_LINEAR_KERNEL_H_
#define INFINI_OPS_MOORE_LINEAR_KERNEL_H_

#include <optional>

#include "base/linear.h"
#include "moore/add/kernel.h"
#include "moore/gemm/mublas.h"

namespace infini::ops {

template <>
class Operator<Linear, Device::Type::kMoore> : public Linear {
 public:
  Operator(const Tensor a, const Tensor b, std::optional<Tensor> bias,
           bool trans_a, bool trans_b, Tensor out)
      : Linear(a, b, bias, trans_a, trans_b, out),
        gemm_(a, b, std::optional<float>{1.0f}, std::optional<float>{0.0f},
              std::optional<int>{static_cast<int>(trans_a)},
              std::optional<int>{static_cast<int>(trans_b)}, out) {
    if (has_bias_) {
      add_.emplace(out, BroadcastBias(*bias, out), out);
    }
  }

  void operator()(const Tensor a, const Tensor b, std::optional<Tensor> bias,
                  bool trans_a, bool trans_b, Tensor out) const override {
    assert(has_bias_ == bias.has_value());

    ConfigureSubOperator(gemm_);
    gemm_(a, b, std::optional<float>{1.0f}, std::optional<float>{0.0f},
          std::optional<int>{static_cast<int>(trans_a)},
          std::optional<int>{static_cast<int>(trans_b)}, out);

    if (has_bias_) {
      auto bias_view = BroadcastBias(*bias, out);
      ConfigureSubOperator(*add_);
      (*add_)(out, bias_view, out);
    }
  }

 private:
  static Tensor BroadcastBias(const Tensor& bias, const Tensor& out) {
    assert(bias.ndim() == 1);
    assert(bias.size(0) == out.size(-1));

    auto shape = out.shape();
    Tensor::Strides strides(out.ndim(), 0);
    strides.back() = bias.stride(0);

    return Tensor{const_cast<void*>(bias.data()), shape, bias.dtype(),
                  bias.device(), strides};
  }

  template <typename Op>
  void ConfigureSubOperator(Op& op) const {
    op.set_handle(handle_);
    op.set_config(config_);
    op.set_stream(stream_);
    op.set_workspace(workspace_);
    op.set_workspace_size_in_bytes(workspace_size_in_bytes_);
  }

  mutable Operator<Gemm, Device::Type::kMoore> gemm_;

  mutable std::optional<Operator<Add, Device::Type::kMoore>> add_;
};

}  // namespace infini::ops

#endif
