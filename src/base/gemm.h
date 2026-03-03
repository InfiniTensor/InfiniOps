#ifndef INFINI_OPS_BASE_GEMM_H_
#define INFINI_OPS_BASE_GEMM_H_

#include <algorithm>
#include <optional>

#include "operator.h"

namespace infini::ops {

class Gemm : public Operator<Gemm> {
 public:
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

  virtual void operator()(void* stream, const Tensor a, const Tensor b,
                          std::optional<float> alpha, std::optional<float> beta,
                          std::optional<int> trans_a,
                          std::optional<int> trans_b, Tensor c) const = 0;

  virtual void operator()(void* stream, const Tensor a, const Tensor b,
                          Tensor c) const {
    return operator()(stream, a, b, std::nullopt, std::nullopt, std::nullopt,
                      std::nullopt, c);
  }

  virtual void operator()(void* stream, const Tensor a, const Tensor b,
                          std::optional<float> alpha, std::optional<float> beta,
                          Tensor c) const {
    return operator()(stream, a, b, alpha, beta, std::nullopt, std::nullopt, c);
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
