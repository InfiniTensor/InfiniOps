#ifndef INFINI_OPS_BASE_ADAM_ACCUMULATE_GRAD_H_
#define INFINI_OPS_BASE_ADAM_ACCUMULATE_GRAD_H_

#include <cassert>
#include <cstdint>

#include "operator.h"

namespace infini::ops {

class AdamAccumulateGrad : public Operator<AdamAccumulateGrad> {
 public:
  AdamAccumulateGrad(const Tensor grad, Tensor param, Tensor m, Tensor v,
                     const double learning_rate, const double beta1,
                     const double beta2, const double eps, const int64_t step)
      : grad_shape_{grad.shape()},
        grad_strides_{grad.strides()},
        grad_type_{grad.dtype()},
        param_shape_{param.shape()},
        param_strides_{param.strides()},
        param_type_{param.dtype()},
        m_shape_{m.shape()},
        m_strides_{m.strides()},
        m_type_{m.dtype()},
        v_shape_{v.shape()},
        v_strides_{v.strides()},
        v_type_{v.dtype()},
        learning_rate_{learning_rate},
        beta1_{beta1},
        beta2_{beta2},
        eps_{eps},
        step_{step},
        device_index_{param.device().index()} {
    assert(grad.shape() == param.shape() && param.shape() == m.shape() &&
           m.shape() == v.shape() &&
           "`AdamAccumulateGrad` requires grad/param/m/v to share shape");
    assert(grad_type_ == param_type_ && param_type_ == m_type_ &&
           m_type_ == v_type_ && "`AdamAccumulateGrad` requires matching dtypes");
    assert(grad.device() == param.device() &&
           "`AdamAccumulateGrad` requires grad/param on the same device");
    assert(0.0 < beta1_ && beta1_ < 1.0 && 0.0 < beta2_ && beta2_ < 1.0 &&
           "`AdamAccumulateGrad` requires beta1/beta2 in (0, 1)");
    assert(step_ >= 1 && "`AdamAccumulateGrad` requires step >= 1");
  }

  virtual void operator()(const Tensor grad, Tensor param, Tensor m, Tensor v,
                          const double learning_rate, const double beta1,
                          const double beta2, const double eps,
                          const int64_t step) const = 0;

 protected:
  Tensor::Shape grad_shape_;
  Tensor::Strides grad_strides_;
  DataType grad_type_;
  Tensor::Shape param_shape_;
  Tensor::Strides param_strides_;
  DataType param_type_;
  Tensor::Shape m_shape_;
  Tensor::Strides m_strides_;
  DataType m_type_;
  Tensor::Shape v_shape_;
  Tensor::Strides v_strides_;
  DataType v_type_;
  double learning_rate_{0.0};
  double beta1_{0.9};
  double beta2_{0.999};
  double eps_{1e-8};
  int64_t step_{1};
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_ADAM_ACCUMULATE_GRAD_H_