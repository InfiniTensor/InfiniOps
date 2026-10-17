#pragma once
#include "base/onnx_gather.h"
namespace infini::ops {
template <Device::Type kDev>
class Operator<OnnxGather, kDev, 8> : public OnnxGather {
 public:
  using OnnxGather::OnnxGather;
  void operator()(const Tensor input, const Tensor index,
                  const int64_t axis, Tensor out) const override;
};
}
