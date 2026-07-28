#ifndef INFINI_OPS_BASE_RANDOM_H_
#define INFINI_OPS_BASE_RANDOM_H_

#include <optional>

#include "generator.h"
#include "operator.h"

namespace infini::ops {

class Random : public Operator<Random> {
 public:
  Random(Tensor input, const int64_t from, const std::optional<int64_t> to,
         const std::optional<Generator> generator)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        from_{from},
        to_{to},
        device_index_{input.device().index()} {}

  Random(Tensor input, const int64_t to,
         const std::optional<Generator> generator)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        device_index_{input.device().index()} {}

  /// \deprecated Use the overload that accepts `generator`. This constructor
  /// will be removed in a future release.
  [[deprecated("Use the overload that accepts `generator` instead.")]]
  Random(Tensor input, const int64_t from, const std::optional<int64_t> to)
      : Random{input, from, to, std::nullopt} {}

  /// \deprecated Use the overload that accepts `generator`. This constructor
  /// will be removed in a future release.
  [[deprecated("Use the overload that accepts `generator` instead.")]]
  Random(Tensor input, const int64_t to)
      : Random{input, to, std::optional<Generator>{}} {}

  virtual void operator()(Tensor input, const int64_t from,
                          const std::optional<int64_t> to,
                          const std::optional<Generator> generator) const = 0;

  virtual void operator()(Tensor input, const int64_t to,
                          const std::optional<Generator> generator) const = 0;

  /// \deprecated Use the overload that accepts `generator`. This overload will
  /// be removed in a future release.
  [[deprecated("Use the overload that accepts `generator` instead.")]]
  void operator()(Tensor input, const int64_t from,
                  const std::optional<int64_t> to) const {
    (*this)(input, from, to, std::nullopt);
  }

  /// \deprecated Use the overload that accepts `generator`. This overload will
  /// be removed in a future release.
  [[deprecated("Use the overload that accepts `generator` instead.")]]
  void operator()(Tensor input, const int64_t to) const {
    (*this)(input, to, std::optional<Generator>{});
  }

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  int64_t from_{};

  std::optional<int64_t> to_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
