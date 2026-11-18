#ifndef INFINI_OPS_BASE_CROSS_ENTROPY_H_
#define INFINI_OPS_BASE_CROSS_ENTROPY_H_

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>

#include "base/cross_entropy_common.h"
#include "operator.h"

namespace infini::ops {

class CrossEntropy : public Operator<CrossEntropy> {
 public:
  static constexpr int64_t kIgnoreIndex = -100;

  CrossEntropy(const Tensor input, const Tensor target,
               std::optional<Tensor> weight, int64_t ignore_index,
               std::string reduction, double label_smoothing, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        target_shape_{target.shape()},
        target_strides_{target.strides()},
        target_type_{target.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {
    ValidateCrossEntropy(input, target, weight, ignore_index, reduction,
                         label_smoothing, out);
  }

  virtual void operator()(const Tensor input, const Tensor target,
                          std::optional<Tensor> weight, int64_t ignore_index,
                          std::string reduction, double label_smoothing,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape target_shape_;

  Tensor::Strides target_strides_;

  DataType target_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_CROSS_ENTROPY_H_
