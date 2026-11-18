#ifndef INFINI_OPS_BASE_CROSS_ENTROPY_COMMON_H_
#define INFINI_OPS_BASE_CROSS_ENTROPY_COMMON_H_

#include <optional>
#include <string>

#include "base/validation.h"

namespace infini::ops {

inline DataType PromoteCrossEntropyType(DataType a, DataType b) {
  if (a == b) return a;
  if (a == DataType::kFloat64 || b == DataType::kFloat64) {
    return DataType::kFloat64;
  }
  return DataType::kFloat32;
}

inline void ValidateCrossEntropy(const Tensor input, const Tensor target,
                                 std::optional<Tensor> weight,
                                 int64_t ignore_index,
                                 const std::string& reduction,
                                 double label_smoothing, const Tensor loss) {
  validation::AssertSameDevice({input, target, loss});

  assert(input.ndim() >= 1 && "`CrossEntropy` requires input rank >= 1.");
  assert(validation::IsFloatingPoint(input.dtype()) &&
         "`CrossEntropy` requires floating-point logits.");
  assert((reduction == "none" || reduction == "mean" || reduction == "sum") &&
         "`CrossEntropy` reduction must be none, mean or sum.");
  assert(label_smoothing >= 0.0 && label_smoothing <= 1.0 &&
         "`CrossEntropy` label_smoothing must be in [0, 1].");
  const auto class_dim = input.ndim() == 1 ? 0 : 1;
  assert(input.size(class_dim) > 0 &&
         "`CrossEntropy` requires a non-empty class dimension.");
  Tensor::Shape loss_shape;
  for (Tensor::Size i = 0; i < input.ndim(); ++i) {
    if (i != class_dim) loss_shape.push_back(input.size(i));
  }
  const bool probability_target = target.shape() == input.shape();

  auto result_type = input.dtype();
  if (probability_target) {
    assert(validation::IsFloatingPoint(target.dtype()) &&
           "`CrossEntropy` probability target must be floating-point.");
    assert(ignore_index < 0 &&
           "`CrossEntropy` probability target does not support ignore_index.");
    result_type = PromoteCrossEntropyType(result_type, target.dtype());
  } else {
    assert((target.dtype() == DataType::kInt64 ||
            target.dtype() == DataType::kInt32 ||
            target.dtype() == DataType::kUInt8) &&
           "`CrossEntropy` class target must be int64, int32 or uint8.");
    assert(
        target.shape() == loss_shape &&
        "`CrossEntropy` class target shape must remove the class dimension.");
  }
  if (weight) {
    validation::AssertSameDevice({input, *weight});
    assert(weight->ndim() == 1 && weight->size(0) == input.size(class_dim) &&
           "`CrossEntropy` weight must have shape (C,).");
    if (probability_target) {
      if (validation::IsFloatingPoint(weight->dtype())) {
        result_type = PromoteCrossEntropyType(result_type, weight->dtype());
      }
    } else {
      assert(weight->dtype() == input.dtype() &&
             "`CrossEntropy` class-target weight dtype must match logits.");
    }
  }
  assert(loss.shape() == (reduction == "none" ? loss_shape : Tensor::Shape{}) &&
         "`CrossEntropy` loss shape does not match reduction.");
  assert(loss.dtype() == result_type && "`CrossEntropy` loss dtype mismatch.");
}

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_CROSS_ENTROPY_COMMON_H_
