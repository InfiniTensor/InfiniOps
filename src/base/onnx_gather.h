#ifndef INFINI_OPS_BASE_ONNX_GATHER_H_
#define INFINI_OPS_BASE_ONNX_GATHER_H_

#include "operator.h"

namespace infini::ops {

// ONNX Gather-13 inserts the entire indices shape at the selected axis.
// The existing Gather instead implements PyTorch's equal-rank gather.
class OnnxGather : public Operator<OnnxGather> {
 public:
  static bool Valid(const Tensor input, const Tensor indices, int64_t axis,
                    const Tensor out) {
    const auto rank = static_cast<int64_t>(input.ndim());
    if (rank == 0 || axis < -rank || axis >= rank ||
        input.device() != indices.device() || input.device() != out.device() ||
        input.dtype() != out.dtype() ||
        (indices.dtype() != DataType::kInt32 &&
         indices.dtype() != DataType::kInt64)) {
      return false;
    }
    if (axis < 0) axis += rank;
    std::vector<Tensor::Size> expected;
    const auto shape = input.shape();
    const auto index_shape = indices.shape();
    expected.insert(expected.end(), shape.begin(), shape.begin() + axis);
    expected.insert(expected.end(), index_shape.begin(), index_shape.end());
    expected.insert(expected.end(), shape.begin() + axis + 1, shape.end());
    return expected.size() == out.ndim() &&
           std::equal(expected.begin(), expected.end(), out.shape().begin());
  }

  OnnxGather(const Tensor input, const Tensor indices, const int64_t axis,
             Tensor out) {
    assert(Valid(input, indices, axis, out) &&
           "Invalid ONNX Gather axis, shape, dtype or device.");
  }

  virtual void operator()(const Tensor input, const Tensor indices,
                          const int64_t axis, Tensor out) const = 0;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_ONNX_GATHER_H_
