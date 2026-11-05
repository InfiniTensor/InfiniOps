#ifndef INFINI_OPS_CAMBRICON_CLIP_KERNEL_H_
#define INFINI_OPS_CAMBRICON_CLIP_KERNEL_H_

#include <limits>

#include "base/clip.h"
#include "native/cambricon/cnnl_utils.h"

namespace infini::ops {

template <>
class Operator<Clip, Device::Type::kCambricon> : public Clip {
 public:
  Operator(const Tensor input, const std::optional<double> min,
           const std::optional<double> max, Tensor out)
      : Clip{input, min, max, out} {
    Init(input, out);
  }

  Operator(const Tensor input, const std::optional<Tensor> min,
           const std::optional<Tensor> max, Tensor out)
      : Clip{input, min, max, out} {
    if (!min || !max || min->numel() != 1 || max->numel() != 1 ||
        min->dtype() != input.dtype() || max->dtype() != input.dtype() ||
        min->device() != input.device() || max->device() != input.device())
      throw std::invalid_argument("Cambricon Clip device bounds must be two same-type scalars");
    Init(input, out);
  }

  void operator()(const Tensor input, const std::optional<double> min,
                  const std::optional<double> max, Tensor out) const override {
    if (!descriptor_) return;
    const auto infinity = std::numeric_limits<float>::infinity();
    const float low = min.value_or(-infinity), high = max.value_or(infinity);
    if (input.dtype() == DataType::kFloat16) {
      const auto low_half = ::infini::rt::Float16::FromFloat(low).bits;
      const auto high_half = ::infini::rt::Float16::FromFloat(high).bits;
      Run(input, CNNL_POINTER_MODE_HOST, &low_half, &high_half, out);
    } else {
      Run(input, CNNL_POINTER_MODE_HOST, &low, &high, out);
    }
  }

  void operator()(const Tensor input, const std::optional<Tensor> min,
                  const std::optional<Tensor> max, Tensor out) const override {
    if (descriptor_) Run(input, CNNL_POINTER_MODE_DEVICE, min->data(), max->data(), out);
  }

 private:
  void Init(const Tensor& input, const Tensor& out) {
    if (input.shape() != out.shape() || input.dtype() != out.dtype() ||
        input.device() != out.device() || !input.IsContiguous() || !out.IsContiguous() ||
        (out.dtype() != DataType::kFloat32 && out.dtype() != DataType::kFloat16))
      throw std::invalid_argument("Cambricon Clip requires matching FP32 or FP16 tensors");
    if (out.numel() == 0) return;
    handle_ = cnnl_utils::CreateHandle();
    descriptor_ = cnnl_utils::MakeTensorDescriptor(
        input.dtype(), Tensor::Shape(input.shape()), Tensor::Strides(input.strides()));
  }

  void Run(const Tensor& input, cnnlPointerMode_t pointer_mode,
           const void* min, const void* max, Tensor out) const {
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)));
    INFINI_OPS_CNNL_CHECK(cnnlClip_v2(handle_.get(), pointer_mode, descriptor_.get(),
        input.data(), min, max, descriptor_.get(), out.data()));
  }

  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor descriptor_;
};

}  // namespace infini::ops

#endif
