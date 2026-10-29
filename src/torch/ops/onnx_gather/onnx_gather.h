#ifndef INFINI_OPS_TORCH_ONNX_GATHER_H_
#define INFINI_OPS_TORCH_ONNX_GATHER_H_

#include "base/onnx_gather.h"

namespace infini::ops {

template <Device::Type kDev>
class AtenOnnxGather : public OnnxGather {
 public:
  using OnnxGather::OnnxGather;

  static bool supports(const Tensor input, const Tensor indices,
                       const int64_t axis, const Tensor out) {
    return input.device().type() == kDev &&
           OnnxGather::Valid(input, indices, axis, out) &&
           (out.numel() == 0 || !out.HasBroadcastDim());
  }

  void operator()(const Tensor input, const Tensor indices, const int64_t axis,
                  Tensor out) const override;
};

template <>
class Operator<OnnxGather, Device::Type::kCpu, 8>
    : public AtenOnnxGather<Device::Type::kCpu> {
 public:
  using AtenOnnxGather<Device::Type::kCpu>::AtenOnnxGather;
};

template <>
class Operator<OnnxGather, Device::Type::kNvidia, 8>
    : public AtenOnnxGather<Device::Type::kNvidia> {
 public:
  using AtenOnnxGather<Device::Type::kNvidia>::AtenOnnxGather;
};

template <>
class Operator<OnnxGather, Device::Type::kCambricon, 8>
    : public AtenOnnxGather<Device::Type::kCambricon> {
 public:
  using AtenOnnxGather<Device::Type::kCambricon>::AtenOnnxGather;
};

template <>
class Operator<OnnxGather, Device::Type::kAscend, 8>
    : public AtenOnnxGather<Device::Type::kAscend> {
 public:
  using AtenOnnxGather<Device::Type::kAscend>::AtenOnnxGather;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_ONNX_GATHER_H_
