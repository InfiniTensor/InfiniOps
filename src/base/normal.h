#ifndef INFINI_OPS_BASE_NORMAL_H_
#define INFINI_OPS_BASE_NORMAL_H_

#include <optional>
#include <vector>

#include "generator.h"
#include "operator.h"

namespace infini::ops {

class Normal : public Operator<Normal> {
 public:
  Normal(const Tensor mean, const double std,
         const std::optional<Generator> generator, Tensor out)
      : mean_shape_{mean.shape()},
        mean_strides_{mean.strides()},
        mean_type_{mean.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        std_{std},
        device_index_{out.device().index()} {}

  Normal(const Tensor mean, const Tensor std,
         const std::optional<Generator> generator, Tensor out)
      : mean_shape_{mean.shape()},
        mean_strides_{mean.strides()},
        mean_type_{mean.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        std_shape_{std.shape()},
        std_strides_{std.strides()},
        std_type_{std.dtype()},
        device_index_{out.device().index()} {}

  Normal(const double mean, const Tensor std,
         const std::optional<Generator> generator, Tensor out)
      : out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        mean_{mean},
        std_shape_{std.shape()},
        std_strides_{std.strides()},
        std_type_{std.dtype()},
        device_index_{out.device().index()} {}

  Normal(const double mean, const double std, const std::vector<int64_t> size,
         const std::optional<Generator> generator, Tensor out)
      : out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        mean_{mean},
        std_{std},
        size_{size},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor mean, const double std,
                          const std::optional<Generator> generator,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor mean, const Tensor std,
                          const std::optional<Generator> generator,
                          Tensor out) const = 0;

  virtual void operator()(const double mean, const Tensor std,
                          const std::optional<Generator> generator,
                          Tensor out) const = 0;

  virtual void operator()(const double mean, const double std,
                          const std::vector<int64_t> size,
                          const std::optional<Generator> generator,
                          Tensor out) const = 0;

  /// \deprecated Use the overload with `generator`. This constructor will be
  /// removed in a future release.
  [[deprecated("Use the overload with `generator` instead.")]]
  Normal(const Tensor mean, const double std, Tensor out)
      : Normal{mean, std, std::nullopt, out} {}

  /// \deprecated Use the overload with `generator`. This constructor will be
  /// removed in a future release.
  [[deprecated("Use the overload with `generator` instead.")]]
  Normal(const Tensor mean, const Tensor std, Tensor out)
      : Normal{mean, std, std::nullopt, out} {}

  /// \deprecated Use the overload with `generator`. This overload will be
  /// removed in a future release.
  [[deprecated("Use the overload with `generator` instead.")]]
  void operator()(const Tensor mean, const double std, Tensor out) const {
    (*this)(mean, std, std::nullopt, out);
  }

  /// \deprecated Use the overload with `generator`. This overload will be
  /// removed in a future release.
  [[deprecated("Use the overload with `generator` instead.")]]
  void operator()(const Tensor mean, const Tensor std, Tensor out) const {
    (*this)(mean, std, std::nullopt, out);
  }

 protected:
  Tensor::Shape mean_shape_;

  Tensor::Strides mean_strides_;

  DataType mean_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double mean_{};

  double std_{};

  std::vector<int64_t> size_{};

  Tensor::Shape std_shape_;

  Tensor::Strides std_strides_;

  DataType std_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
