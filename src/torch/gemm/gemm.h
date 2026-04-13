#ifndef INFINI_OPS_TORCH_GEMM_H_
#define INFINI_OPS_TORCH_GEMM_H_

#include "base/gemm.h"
#include "torch/gemm/registry.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<Gemm, kDev, 2> : public Gemm {
 public:
  Operator(const Tensor a, const Tensor b, std::optional<float> alpha,
           std::optional<float> beta, std::optional<int> trans_a,
           std::optional<int> trans_b, Tensor c)
      : Gemm{a, b, alpha, beta, trans_a, trans_b, c},
        a_shape_{a.shape()},
        b_shape_{b.shape()},
        c_shape_{c.shape()},
        device_index_{c.device().index()} {}

  using Gemm::operator();

  void operator()(const Tensor a, const Tensor b, std::optional<float> alpha,
                  std::optional<float> beta, std::optional<int> trans_a,
                  std::optional<int> trans_b, Tensor c) const override {
    // Use stored metadata instead of parameter tensors, which may be
    // moved-from by the `call()` dispatch path (see `operator.h`).
    auto at_a = ToAtenTensor<kDev>(const_cast<void*>(a.data()), a_shape_,
                                   a_strides_, a_type_, device_index_);
    auto at_b = ToAtenTensor<kDev>(const_cast<void*>(b.data()), b_shape_,
                                   b_strides_, b_type_, device_index_);
    auto at_c = ToAtenTensor<kDev>(c.data(), c_shape_, c_strides_, c_type_,
                                   device_index_);

    auto alpha_val = alpha.value_or(alpha_);
    auto beta_val = beta.value_or(beta_);

    if (trans_a.value_or(trans_a_)) {
      at_a = at_a.transpose(-2, -1);
    }

    if (trans_b.value_or(trans_b_)) {
      at_b = at_b.transpose(-2, -1);
    }

    if (at_a.dim() == 2) {
      at::addmm_out(at_c, at_c, at_a, at_b, beta_val, alpha_val);
    } else {
      at::baddbmm_out(at_c, at_c, at_a, at_b, beta_val, alpha_val);
    }
  }

 private:
  Tensor::Shape a_shape_;

  Tensor::Shape b_shape_;

  Tensor::Shape c_shape_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
