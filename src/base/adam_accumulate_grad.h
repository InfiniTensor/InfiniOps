#ifndef INFINI_OPS_BASE_ADAM_ACCUMULATE_GRAD_H_
#define INFINI_OPS_BASE_ADAM_ACCUMULATE_GRAD_H_

#include <cassert>
#include <cstdint>
#include <initializer_list>

#include "operator.h"

namespace infini::ops {

class AdamAccumulateGrad : public Operator<AdamAccumulateGrad> {
 public:
  AdamAccumulateGrad(const Tensor grad, const double learning_rate,
                     const double beta1, const double beta2, const double eps,
                     const int64_t step, Tensor param, Tensor m, Tensor v)
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
    AssertTrainingDevices({grad, param, m, v});
    assert(grad.shape() == param.shape() && param.shape() == m.shape() &&
           m.shape() == v.shape() &&
           "`AdamAccumulateGrad` requires grad/param/m/v to share shape");
    assert(IsTrainingFloat(param_type_) && grad_type_ == param_type_ &&
           "Adam parameters and gradients require matching floating dtypes.");
    assert(m_type_ == v_type_ &&
           (m_type_ == param_type_ || (param_type_ != DataType::kFloat64 &&
                                       m_type_ == DataType::kFloat32)) &&
           "Adam moments must match the parameter dtype or use float32.");
    assert(grad.device() == param.device() &&
           "`AdamAccumulateGrad` requires grad/param on the same device");
    assert(0.0 <= beta1_ && beta1_ < 1.0 && 0.0 <= beta2_ && beta2_ < 1.0 &&
           "Adam beta values must be in [0, 1).");
    assert(learning_rate_ >= 0.0 && eps_ > 0.0 &&
           "Adam requires non-negative learning rate and positive epsilon.");
    assert(step_ >= 1 && "`AdamAccumulateGrad` requires step >= 1");
  }

  [[deprecated("Use grad, attributes, param, m, v argument order.")]]
  AdamAccumulateGrad(const Tensor grad, Tensor param, Tensor m, Tensor v,
                     const double learning_rate, const double beta1,
                     const double beta2, const double eps, const int64_t step)
      : AdamAccumulateGrad{grad, learning_rate, beta1, beta2, eps,
                           step, param,         m,     v} {}

  [[deprecated("Use grad, attributes, param, m, v argument order.")]]
  virtual void operator()(const Tensor grad, Tensor param, Tensor m, Tensor v,
                          const double learning_rate, const double beta1,
                          const double beta2, const double eps,
                          const int64_t step) const {
    operator()(grad, learning_rate, beta1, beta2, eps, step, param, m, v);
  }

  virtual void operator()(const Tensor grad, const double learning_rate,
                          const double beta1, const double beta2,
                          const double eps, const int64_t step, Tensor param,
                          Tensor m, Tensor v) const = 0;

 protected:
  static void AssertTrainingDevices(std::initializer_list<Tensor> tensors) {
    if (tensors.size() == 0) return;
    const auto device = tensors.begin()->device();
    for (const auto& tensor : tensors) {
      assert(tensor.device() == device &&
             "Training tensors must share a device type and index.");
    }
  }

  static bool IsTrainingFloat(DataType dtype) {
    return dtype == DataType::kFloat16 || dtype == DataType::kBFloat16 ||
           dtype == DataType::kFloat32 || dtype == DataType::kFloat64;
  }

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
