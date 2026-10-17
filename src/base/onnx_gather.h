#pragma once
#include "operator.h"

namespace infini::ops {
// ONNX Gather inserts the complete index shape at axis, unlike ATen gather.
class OnnxGather : public Operator<OnnxGather> {
 public:
  OnnxGather(const Tensor input, const Tensor index, const int64_t axis, Tensor out) {}
  virtual void operator()(const Tensor input, const Tensor index,
                          const int64_t axis, Tensor out) const = 0;
};
}
