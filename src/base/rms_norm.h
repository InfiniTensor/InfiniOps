#ifndef INFINI_OPS_BASE_RMS_NORM_H_
#define INFINI_OPS_BASE_RMS_NORM_H_

#include <cstddef>
#include <vector>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class RmsNorm : public Operator<RmsNorm> {
 public:
  static bool Valid(const Tensor input, const Tensor weight, const Tensor out) {
    return input.ndim() != 0 && input.shape() == out.shape() &&
           input.size(-1) != 0 && weight.ndim() == 1 &&
           weight.size(0) == input.size(-1) && input.dtype() == out.dtype() &&
           input.device() == out.device() && input.device() == weight.device();
  }

  static bool supports(const Tensor input, const Tensor weight, float,
                       const Tensor out) {
    return Valid(input, weight, out);
  }

  static bool supports(const Tensor input, const Tensor weight,
                       const Tensor out) {
    return Valid(input, weight, out);
  }

  RmsNorm(const Tensor input, const Tensor weight, float eps, Tensor out)
      : input_shape_{input.shape()},
        out_shape_{out.shape()},
        input_strides_{input.strides()},
        out_strides_{out.strides()},
        eps_{eps} {
    assert(Valid(input, weight, out) &&
           "Invalid RmsNorm shape, dtype or device");
    dim_ = out.size(-1);
    ndim_ = out.ndim();
    batch_size_ = ndim_ >= 3 ? out.size(-3) : (ndim_ == 2 ? out.size(0) : 1);
    nhead_ = ndim_ >= 3 ? out.size(-2) : 1;
  }

  RmsNorm(const Tensor input, const Tensor weight, Tensor out)
      : RmsNorm{input, weight, 1e-6f, out} {}

  // TODO: Type of `eps` should be `std::optional<float>` instead of `float`.
  virtual void operator()(const Tensor input, const Tensor weight, float eps,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor input, const Tensor weight,
                          Tensor out) const {
    return operator()(input, weight, eps_, out);
  }

 protected:
  // CPU/CUDA row kernels use two outer dimensions and unit inner strides.
  // Other providers (e.g. NineToothed) support different ranks and layouts.
  static bool NativeSupports(const Tensor input, const Tensor weight,
                             const Tensor out) {
    return Valid(input, weight, out) &&
           (input.ndim() == 2 || input.ndim() == 3) &&
           input.strides().back() == 1 && out.strides().back() == 1 &&
           weight.strides().back() == 1;
  }

  Tensor::Shape input_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides input_strides_;

  Tensor::Strides out_strides_;

  float eps_{1e-6f};

  Tensor::Size dim_{0};

  Tensor::Size ndim_{0};

  Tensor::Size batch_size_{0};

  Tensor::Size nhead_{1};
};

}  // namespace infini::ops

#endif
