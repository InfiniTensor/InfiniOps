#ifndef INFINI_OPS_TORCH_ONNX_GATHER_H_
#define INFINI_OPS_TORCH_ONNX_GATHER_H_
#include "base/onnx_gather.h"
#include "torch/tensor_bounds.h"
namespace infini::ops {
template <Device::Type kDev>
class Operator<OnnxGather, kDev, 8> : public OnnxGather {
 public:
  static bool supports(const Tensor input, const Tensor index,
                       const int64_t axis, Tensor out) {
    return input.device().type() == kDev &&
           OnnxGather::Valid(input, index, axis, out) &&
           torch_detail::Disjoint(input, out) &&
           torch_detail::Disjoint(index, out);
  }
  Operator(const Tensor input, const Tensor index, const int64_t axis,
           Tensor out)
      : OnnxGather{input, index, axis, out} {}
  void operator()(const Tensor input, const Tensor index, const int64_t axis,
                  Tensor out) const override;
};
}  // namespace infini::ops
#endif
