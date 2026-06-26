#ifndef INFINI_OPS_BASE_GEMM_H_
#define INFINI_OPS_BASE_GEMM_H_

#include <algorithm>
#include <optional>

#include "operator.h"

namespace infini::ops {

class Gemm : public Operator<Gemm> {
 public:
  using Operator<Gemm>::Call;

  Gemm(const Tensor a, const Tensor b, std::optional<float> alpha,
       std::optional<float> beta, std::optional<int> trans_a,
       std::optional<int> trans_b, Tensor c)
      : alpha_{alpha.value_or(1.0)},
        beta_{beta.value_or(1.0)},
        trans_a_{static_cast<bool>(trans_a.value_or(false))},
        trans_b_{static_cast<bool>(trans_b.value_or(false))},
        m_{c.size(-2)},
        n_{c.size(-1)},
        k_{trans_a_ ? a.size(-2) : a.size(-1)},
        a_type_{a.dtype()},
        b_type_{b.dtype()},
        c_type_{c.dtype()},
        a_strides_{a.strides()},
        b_strides_{b.strides()},
        c_strides_{c.strides()},
        lda_{std::max(a.stride(-2), a.stride(-1))},
        ldb_{std::max(b.stride(-2), b.stride(-1))},
        ldc_{std::max(c.stride(-2), c.stride(-1))},
        batch_count_{c.strides().size() > 2 ? c.size(-3) : 1},
        batch_stride_a_{a.strides().size() > 2 ? a.stride(-3) : 0},
        batch_stride_b_{b.strides().size() > 2 ? b.stride(-3) : 0},
        batch_stride_c_{c.strides().size() > 2 ? c.stride(-3) : 0} {
    // TODO: Check constraints.
  }

  Gemm(const Tensor a, const Tensor b, Tensor c)
      : Gemm{a, b, std::nullopt, std::nullopt, std::nullopt, std::nullopt, c} {}

  virtual void operator()(const Tensor a, const Tensor b,
                          std::optional<float> alpha, std::optional<float> beta,
                          std::optional<int> trans_a,
                          std::optional<int> trans_b, Tensor c) const = 0;

  virtual void operator()(const Tensor a, const Tensor b, Tensor c) const {
    return operator()(a, b, std::nullopt, std::nullopt, std::nullopt,
                      std::nullopt, c);
  }

  virtual void operator()(const Tensor a, const Tensor b,
                          std::optional<float> alpha, std::optional<float> beta,
                          Tensor c) const {
    return operator()(a, b, alpha, beta, std::nullopt, std::nullopt, c);
  }

  template <typename TensorLike>
  static auto Call(const TensorLike& a, const TensorLike& b) {
    return Call(a, b, false, false);
  }

  template <typename TensorLike>
  static auto Call(const TensorLike& a, const TensorLike& b, bool trans_a,
                   bool trans_b) {
    Tensor::Shape c_shape{
        trans_a ? a.shape()[a.shape().size() - 1]
                : a.shape()[a.shape().size() - 2],
        trans_b ? b.shape()[b.shape().size() - 2]
                : b.shape()[b.shape().size() - 1],
    };
    auto c = TensorLike::Empty(c_shape, a.dtype(), a.device());
    Tensor a_view{const_cast<void*>(static_cast<const void*>(a.data())),
                  a.shape(), a.dtype(), a.device(), a.strides()};
    Tensor b_view{const_cast<void*>(static_cast<const void*>(b.data())),
                  b.shape(), b.dtype(), b.device(), b.strides()};
    Tensor c_view{c.data(), c.shape(), c.dtype(), c.device(), c.strides()};
    Gemm::Call(a_view, b_view, std::optional<float>{1.0f},
               std::optional<float>{0.0f},
               std::optional<int>{static_cast<int>(trans_a)},
               std::optional<int>{static_cast<int>(trans_b)}, c_view);
    return c;
  }

 protected:
  float alpha_{1.0};

  float beta_{1.0};

  bool trans_a_{false};

  bool trans_b_{false};

  Tensor::Size m_{0};

  Tensor::Size n_{0};

  Tensor::Size k_{0};

  const DataType a_type_;

  const DataType b_type_;

  const DataType c_type_;

  Tensor::Strides a_strides_;

  Tensor::Strides b_strides_;

  Tensor::Strides c_strides_;

  Tensor::Stride lda_{0};

  Tensor::Stride ldb_{0};

  Tensor::Stride ldc_{0};

  Tensor::Size batch_count_{1};

  Tensor::Stride batch_stride_a_{0};

  Tensor::Stride batch_stride_b_{0};

  Tensor::Stride batch_stride_c_{0};
};

}  // namespace infini::ops

#endif
