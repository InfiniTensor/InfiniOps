#ifndef INFINI_OPS_BASE_DIV_INPLACE_H_
#define INFINI_OPS_BASE_DIV_INPLACE_H_

#include <optional>
#include <string>

#include "operator.h"

namespace infini::ops {

class DivInplace : public Operator<DivInplace> {
 public:
  DivInplace(Tensor input, const Tensor other)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        other_shape_{other.shape()},
        other_strides_{other.strides()},
        other_type_{other.dtype()},
        device_index_{input.device().index()} {}

  DivInplace(Tensor input, const Tensor other,
             const std::optional<std::string> rounding_mode)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        other_shape_{other.shape()},
        other_strides_{other.strides()},
        other_type_{other.dtype()},
        rounding_mode_{rounding_mode},
        device_index_{input.device().index()} {}

  DivInplace(Tensor input, const double other)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        other_{other},
        device_index_{input.device().index()} {}

  DivInplace(Tensor input, const double other,
             const std::optional<std::string> rounding_mode)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        rounding_mode_{rounding_mode},
        other_{other},
        device_index_{input.device().index()} {}

  virtual void operator()(Tensor input, const Tensor other) const = 0;

  virtual void operator()(
      Tensor input, const Tensor other,
      const std::optional<std::string> rounding_mode) const = 0;

  virtual void operator()(Tensor input, const double other) const = 0;

  virtual void operator()(
      Tensor input, const double other,
      const std::optional<std::string> rounding_mode) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape other_shape_;

  Tensor::Strides other_strides_;

  DataType other_type_;

  std::optional<std::string> rounding_mode_{};

  double other_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
