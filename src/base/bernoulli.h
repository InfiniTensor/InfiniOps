#ifndef INFINI_OPS_BASE_BERNOULLI_H_
#define INFINI_OPS_BASE_BERNOULLI_H_

#include <cassert>
#include <optional>

#include "generator.h"
#include "operator.h"

namespace infini::ops {

class Bernoulli : public Operator<Bernoulli> {
 public:
  Bernoulli(const Tensor input, const std::optional<Generator> generator,
            Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  /// \deprecated Use `Bernoulli(input, generator, out)`. This constructor will
  /// be removed in a future release.
  [[deprecated("Use the `(input, generator, out)` overload instead.")]]
  Bernoulli(const Tensor input, Tensor out)
      : Bernoulli{input, std::nullopt, out} {}

  Bernoulli(Tensor input, const double p)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        p_{p},
        device_index_{input.device().index()} {}

  virtual void operator()(const Tensor input,
                          const std::optional<Generator> generator,
                          Tensor out) const = 0;

  /// \deprecated Use `operator()(input, generator, out)`. This overload will
  /// be removed in a future release.
  [[deprecated("Use `operator()(input, generator, out)` instead.")]]
  void operator()(const Tensor input, Tensor out) const {
    return operator()(input, std::nullopt, out);
  }

  virtual void operator()(Tensor input, const double p) const {
    (void)input;
    (void)p;
    assert(false &&
           "The scalar-probability Bernoulli overload is not implemented");
  }

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double p_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
