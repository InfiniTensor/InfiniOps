#ifndef INFINI_OPS_BASE_SCALED_MM_H_
#define INFINI_OPS_BASE_SCALED_MM_H_

#include <cassert>
#include <limits>
#include <optional>

#include "operator.h"

namespace infini::ops {

class ScaledMm : public Operator<ScaledMm> {
 public:
  ScaledMm(const Tensor a, const Tensor b, const Tensor scale_a,
           const Tensor scale_b, std::optional<Tensor> bias, Tensor out)
      : m_{a.ndim() == 2 ? a.size(0) : 0},
        n_{b.ndim() == 2 ? b.size(1) : 0},
        k_{a.ndim() == 2 ? a.size(1) : 0},
        lda_{a.ndim() == 2 ? a.stride(0) : 0},
        ldb_{b.ndim() == 2 ? b.stride(1) : 0},
        ldo_{out.ndim() == 2 ? out.stride(0) : 0},
        scale_a_size_{scale_a.numel()},
        scale_b_size_{scale_b.numel()},
        out_dtype_{out.dtype()} {
    assert(a.ndim() == 2 && b.ndim() == 2 && out.ndim() == 2 &&
           "`ScaledMm` requires 2D matrices");
    assert(a.dtype() == DataType::kInt8 && b.dtype() == DataType::kInt8 &&
           "`ScaledMm` requires int8 matrix inputs");
    assert((out_dtype_ == DataType::kFloat16 ||
            out_dtype_ == DataType::kBFloat16) &&
           "`ScaledMm` requires float16 or bfloat16 output");
    assert(a.size(1) == b.size(0) && out.size(0) == a.size(0) &&
           out.size(1) == b.size(1) &&
           "`ScaledMm` matrix shapes are incompatible");
    assert(m_ > 0 && n_ > 0 && k_ > 0 &&
           "`ScaledMm` requires non-empty matrices");
    assert(a.stride(1) == 1 && b.stride(0) == 1 && out.stride(1) == 1 &&
           "`ScaledMm` requires row-major `a` and `out` and column-major `b`");
    assert(lda_ >= k_ && ldb_ >= k_ && ldo_ >= n_ &&
           "`ScaledMm` matrix strides must cover their logical dimensions");
    assert(k_ % 16 == 0 && n_ % 16 == 0 && lda_ % 16 == 0 && ldb_ % 16 == 0 &&
           ldo_ % 16 == 0 &&
           "`ScaledMm` requires 16-element aligned matrix dimensions");
    assert(scale_a.dtype() == DataType::kFloat32 &&
           scale_b.dtype() == DataType::kFloat32 &&
           "`ScaledMm` requires float32 scales");
    assert(scale_a.IsContiguous() && scale_b.IsContiguous() &&
           "`ScaledMm` requires contiguous scales");
    const auto scale_a_is_per_token{
        scale_a.ndim() == 2 && scale_a.size(0) == m_ && scale_a.size(1) == 1};
    const auto scale_b_is_per_channel{
        scale_b.ndim() == 2 && scale_b.size(0) == 1 && scale_b.size(1) == n_};
    assert((scale_a_size_ == 1 || scale_a_is_per_token) &&
           (scale_b_size_ == 1 || scale_b_is_per_channel) &&
           "`ScaledMm` scales must be scalar, per-token, or per-channel");
    const auto same_device_as_a = [&](const Tensor tensor) {
      return tensor.device().type() == a.device().type() &&
             tensor.device().index() == a.device().index();
    };
    assert(same_device_as_a(b) && same_device_as_a(scale_a) &&
           same_device_as_a(scale_b) && same_device_as_a(out) &&
           "`ScaledMm` tensors must be on the same device");
    assert(m_ <= std::numeric_limits<int>::max() &&
           n_ <= std::numeric_limits<int>::max() &&
           k_ <= std::numeric_limits<int>::max() &&
           "`ScaledMm` matrix dimensions exceed CUTLASS limits");

    if (bias) {
      assert(bias->ndim() == 1 && bias->numel() == n_ &&
             "`ScaledMm` bias must have shape `[n]`");
      assert(bias->dtype() == out_dtype_ && bias->IsContiguous() &&
             "`ScaledMm` bias must match the output dtype and be contiguous");
      assert(same_device_as_a(*bias) &&
             "`ScaledMm` bias must be on the output device");
    }
  }

  virtual void operator()(const Tensor a, const Tensor b, const Tensor scale_a,
                          const Tensor scale_b, std::optional<Tensor> bias,
                          Tensor out) const = 0;

 protected:
  Tensor::Size m_{0};

  Tensor::Size n_{0};

  Tensor::Size k_{0};

  Tensor::Stride lda_{0};

  Tensor::Stride ldb_{0};

  Tensor::Stride ldo_{0};

  Tensor::Size scale_a_size_{0};

  Tensor::Size scale_b_size_{0};

  DataType out_dtype_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_SCALED_MM_H_
