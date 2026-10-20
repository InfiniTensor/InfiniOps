#ifndef INFINI_OPS_BASE_ONNX_GATHER_H_
#define INFINI_OPS_BASE_ONNX_GATHER_H_

#include <algorithm>
#include <cassert>
#include <vector>

#include "operator.h"

namespace infini::ops {
// ONNX Gather inserts the entire index shape at axis. ATen gather instead
// requires an index tensor with the same rank as the input.
class OnnxGather : public Operator<OnnxGather> {
 public:
  static constexpr bool kAutomaticDefault = true;

  static bool Valid(const Tensor input, const Tensor index, int64_t axis,
                    const Tensor out) {
    const auto rank = static_cast<int64_t>(input.ndim());
    if (rank == 0 || axis < -rank || axis >= rank ||
        input.device() != index.device() || input.device() != out.device() ||
        input.dtype() != out.dtype() ||
        (index.dtype() != DataType::kInt32 &&
         index.dtype() != DataType::kInt64))
      return false;
    if (axis < 0) axis += rank;
    std::vector<Tensor::Size> expected;
    const auto shape = input.shape();
    const auto index_shape = index.shape();
    expected.insert(expected.end(), shape.begin(), shape.begin() + axis);
    expected.insert(expected.end(), index_shape.begin(), index_shape.end());
    expected.insert(expected.end(), shape.begin() + axis + 1, shape.end());
    return expected.size() == out.ndim() &&
           std::equal(expected.begin(), expected.end(), out.shape().begin());
  }

  OnnxGather(const Tensor input, const Tensor index, const int64_t axis,
             Tensor out) {
    assert(Valid(input, index, axis, out) &&
           "ONNX Gather requires a valid axis, shape, dtype and device.");
  }
  virtual void operator()(const Tensor input, const Tensor index,
                          const int64_t axis, Tensor out) const = 0;
};
}  // namespace infini::ops
#endif
