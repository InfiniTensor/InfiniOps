#ifndef INFINI_OPS_BASE_GEMM_H_
#define INFINI_OPS_BASE_GEMM_H_

#include <algorithm>
#include <cassert>
#include <optional>

#include "operator.h"

namespace infini::ops {

class Gemm : public Operator<Gemm> {
 public:
  Gemm(const Tensor a, const Tensor b, const std::optional<Tensor> c,
       std::optional<float> alpha, std::optional<float> beta,
       std::optional<int> trans_a, std::optional<int> trans_b, Tensor y)
      : alpha_{alpha.value_or(1.0)},
        beta_{beta.value_or(1.0)},
        trans_a_{static_cast<bool>(trans_a.value_or(false))},
        trans_b_{static_cast<bool>(trans_b.value_or(false))},
        m_{y.size(-2)},
        n_{y.size(-1)},
        k_{trans_a_ ? a.size(-2) : a.size(-1)},
        a_type_{a.dtype()},
        b_type_{b.dtype()},
        y_type_{y.dtype()},
        a_strides_{a.strides()},
        b_strides_{b.strides()},
        c_shape_{c ? Tensor::Shape{c->shape()} : Tensor::Shape{}},
        c_strides_{c ? Tensor::Strides{c->strides()} : Tensor::Strides{}},
        c_broadcast_strides_{c ? BroadcastStrides(*c, y)
                               : Tensor::Strides(y.ndim(), 0)},
        y_shape_{y.shape()},
        y_strides_{y.strides()},
        lda_{std::max(a.stride(-2), a.stride(-1))},
        ldb_{std::max(b.stride(-2), b.stride(-1))},
        ldy_{std::max(y.stride(-2), y.stride(-1))},
        batch_count_{y.strides().size() > 2 ? y.size(-3) : 1},
        batch_stride_a_{a.strides().size() > 2 ? a.stride(-3) : 0},
        batch_stride_b_{b.strides().size() > 2 ? b.stride(-3) : 0},
        batch_stride_y_{y.strides().size() > 2 ? y.stride(-3) : 0} {
    assert(a.dtype() == b.dtype() && a.dtype() == y.dtype() &&
           (!c || c->dtype() == y.dtype()) &&
           "operator `Gemm` requires A, B, C, and Y to have the same dtype");
    assert((!c || c->data() != y.data()) &&
           "operator `Gemm` does not support C/Y aliasing");
  }

  Gemm(const Tensor a, const Tensor b, Tensor y)
      : Gemm{a,
             b,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             y} {}

  virtual void operator()(const Tensor a, const Tensor b,
                          const std::optional<Tensor> c,
                          std::optional<float> alpha, std::optional<float> beta,
                          std::optional<int> trans_a,
                          std::optional<int> trans_b, Tensor y) const = 0;

  virtual void operator()(const Tensor a, const Tensor b, Tensor y) const {
    return operator()(a, b, std::nullopt, std::nullopt, std::nullopt,
                      std::nullopt, std::nullopt, y);
  }

  template <typename TensorLike>
  static auto MakeReturnValue(const TensorLike& a, const TensorLike& b) {
    Tensor::Shape y_shape{a.shape()[a.shape().size() - 2],
                          b.shape()[b.shape().size() - 1]};
    return TensorLike::Empty(y_shape, a.dtype(), a.device());
  }

 protected:
  static Tensor::Strides BroadcastStrides(const Tensor input,
                                          const Tensor out) {
    assert(input.ndim() <= out.ndim() &&
           "operator `Gemm` C rank must not exceed Y rank");
    Tensor::Strides strides(out.ndim(), 0);
    const auto offset = out.ndim() - input.ndim();

    for (Tensor::Size i = 0; i < input.ndim(); ++i) {
      const auto out_dim = i + offset;
      assert((input.size(i) == 1 || input.size(i) == out.size(out_dim)) &&
             "operator `Gemm` C shape is not broadcast-compatible with Y");
      strides[out_dim] = input.size(i) == 1 ? 0 : input.stride(i);
    }

    return strides;
  }

  float EffectiveBeta(const std::optional<Tensor>& c,
                      std::optional<float> beta) const {
    return c ? beta.value_or(beta_) : 0.0F;
  }

  float alpha_{1.0};

  float beta_{1.0};

  bool trans_a_{false};

  bool trans_b_{false};

  Tensor::Size m_{0};

  Tensor::Size n_{0};

  Tensor::Size k_{0};

  const DataType a_type_;

  const DataType b_type_;

  const DataType y_type_;

  Tensor::Strides a_strides_;

  Tensor::Strides b_strides_;

  Tensor::Shape c_shape_;

  Tensor::Strides c_strides_;

  Tensor::Strides c_broadcast_strides_;

  Tensor::Shape y_shape_;

  Tensor::Strides y_strides_;

  Tensor::Stride lda_{0};

  Tensor::Stride ldb_{0};

  Tensor::Stride ldy_{0};

  Tensor::Size batch_count_{1};

  Tensor::Stride batch_stride_a_{0};

  Tensor::Stride batch_stride_b_{0};

  Tensor::Stride batch_stride_y_{0};
};

}  // namespace infini::ops

#endif
