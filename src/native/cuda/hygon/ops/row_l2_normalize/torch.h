#pragma once
#include <cmath>

#include "base/row_l2_normalize.h"
#include "native/cuda/nvidia/ops/binary_view_capture.h"
namespace infini::ops {
template <>
class Operator<RowL2Normalize, Device::Type::kHygon, 8>
    : public RowL2Normalize {
 public:
  static bool supports(const Tensor input, double eps, Tensor out) {
    using namespace nvidia_binary_view;
    size_t input_bytes = 0, out_bytes = 0;
    return input.device().type() == Device::Type::kHygon &&
           input.device() == out.device() && input.dtype() == out.dtype() &&
           input.shape() == out.shape() && input.ndim() == 2 &&
           (input.dtype() == DataType::kFloat32 ||
            input.dtype() == DataType::kFloat16 ||
            input.dtype() == DataType::kBFloat16) &&
           std::isfinite(eps) && eps >= 0 && NonOverlappingOutput(out) &&
           SpanBytes(input, &input_bytes) && SpanBytes(out, &out_bytes) &&
           (!Overlaps(input, input_bytes, out, out_bytes) ||
            (input.data() == out.data() && input.strides() == out.strides()));
  }
  using RowL2Normalize::RowL2Normalize;
  void operator()(const Tensor input, double eps, Tensor out) const override;
};
}  // namespace infini::ops
