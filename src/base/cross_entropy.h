#ifndef INFINI_OPS_BASE_CROSS_ENTROPY_H_
#define INFINI_OPS_BASE_CROSS_ENTROPY_H_

#include <cassert>
#include <cstdint>
#include <initializer_list>

#include "operator.h"

namespace infini::ops {

class CrossEntropy : public Operator<CrossEntropy> {
 public:
  static constexpr int64_t kIgnoreIndex = -100;

  CrossEntropy(const Tensor input, const Tensor target, Tensor out)
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
    AssertTrainingDevices({input, target, out});
    assert(input.ndim() >= 2 && "`CrossEntropy` requires input rank >= 2");
    assert(input.device() == target.device() &&
           target.device() == out.device() &&
           "`CrossEntropy` requires all tensors on the same device");
    assert((input_type_ == DataType::kFloat16 ||
            input_type_ == DataType::kBFloat16 ||
            input_type_ == DataType::kFloat32 ||
            input_type_ == DataType::kFloat64) &&
           "`CrossEntropy` requires floating-point logits");
    assert((target_type_ == DataType::kInt64 ||
            target_type_ == DataType::kInt32 ||
            target_type_ == DataType::kUInt8) &&
           "`CrossEntropy` supports int64/int32/uint8 targets");
    assert(target.ndim() == input.ndim() - 1 &&
           "`CrossEntropy` target rank must be input rank - 1");
    for (Tensor::Size i = 0; i < target.ndim(); ++i) {
      const auto input_dim = i == 0 ? 0 : i + 1;
      assert(target.size(i) == input.size(input_dim) &&
             "`CrossEntropy` target shape mismatch");
    }
    assert(out.ndim() == 0 && "`CrossEntropy` output must be a 0-dim scalar");
    assert(out_type_ == input_type_ &&
           "`CrossEntropy` output dtype must match input");
  }

  virtual void operator()(const Tensor input, const Tensor target,
                          Tensor out) const = 0;

 protected:
  static void AssertTrainingDevices(std::initializer_list<Tensor> tensors) {
    if (tensors.size() == 0) return;
    const auto device = tensors.begin()->device();
    for (const auto& tensor : tensors) {
      assert(tensor.device() == device &&
             "Training tensors must share a device type and index.");
    }
  }

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
