#ifndef INFINI_OPS_BASE_GEMM_H_
#define INFINI_OPS_BASE_GEMM_H_

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
        m_{c.size(0)},
        n_{c.size(1)},
        k_{trans_a_ ? a.size(0) : a.size(1)},
        a_type_{a.dtype()},
        b_type_{b.dtype()},
        c_type_{c.dtype()},
        a_strides_{a.strides()},
        b_strides_{b.strides()},
        c_strides_{c.strides()} {
    // TODO: Check constraints.
  }

  virtual void operator()(void* stream, const void* a, const void* b,
                          void* c) const = 0;

 protected:
  float alpha_{1.0};

  float beta_{1.0};

  bool trans_a_{false};

  bool trans_b_{false};

  Tensor::Size m_{0};

  Tensor::Size n_{0};

  Tensor::Size k_{0};

  const DataType& a_type_;

  const DataType& b_type_;

  const DataType& c_type_;

  Tensor::Strides a_strides_;

  Tensor::Strides b_strides_;

  Tensor::Strides c_strides_;
};

}  // namespace infini::ops

#endif
